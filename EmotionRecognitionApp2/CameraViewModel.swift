import SwiftUI
import AVFoundation
import Vision
import CoreML
import CoreImage
import CoreImage.CIFilterBuiltins
import UIKit

class CameraViewModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var emotion: String = "알 수 없음"
    
    public let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    
    private let emotionModel: ViT_EmotionDetection_Converted
    private let context = CIContext() // CIContext를 클래스 수준에서 생성해 매번 재사용
    
    private let visionQueue = DispatchQueue(label: "vision.queue")
    
    override init() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly // Metal 관련 문제를 우회하기 위해 CPU 전용 사용
            self.emotionModel = try ViT_EmotionDetection_Converted(configuration: config)
        } catch {
            fatalError("모델을 로드할 수 없습니다: \(error)")
        }
        
        super.init()
        checkPermissions()
        configureSession()
    }
    
    func checkPermissions() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            break
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if !granted {
                    print("카메라 접근 권한이 필요합니다.")
                }
            }
        default:
            print("카메라 접근 권한이 필요합니다.")
        }
    }
    
    func configureSession() {
        sessionQueue.async {
            self.session.beginConfiguration()
            self.session.sessionPreset = .high
            
            guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                print("후면 카메라를 찾을 수 없습니다.")
                return
            }
            
            do {
                let input = try AVCaptureDeviceInput(device: camera)
                if self.session.canAddInput(input) {
                    self.session.addInput(input)
                }
            } catch {
                print("카메라 입력을 추가할 수 없습니다: \(error)")
                return
            }
            
            let videoOutput = AVCaptureVideoDataOutput()
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "video.output.queue"))
            videoOutput.alwaysDiscardsLateVideoFrames = true
            if self.session.canAddOutput(videoOutput) {
                self.session.addOutput(videoOutput)
            }
            
            self.session.commitConfiguration()
            self.session.startRunning()
            print("Camera session configured and started.")
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to get pixel buffer from sample buffer.")
            return
        }
        detectEmotion(on: pixelBuffer)
    }

    func detectEmotion(on pixelBuffer: CVPixelBuffer) {
        let deviceOrientation = UIDevice.current.orientation
        let cgImagePropertyOrientation: CGImagePropertyOrientation = {
            switch deviceOrientation {
            case .portrait: return .up
            case .portraitUpsideDown: return .down
            case .landscapeLeft: return .left
            case .landscapeRight: return .right
            default: return .up
            }
        }()
        
        let request = VNDetectFaceRectanglesRequest { [weak self] (request, error) in
            self?.handleFaceDetection(request: request, error: error, pixelBuffer: pixelBuffer)
        }
        
        let requestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: cgImagePropertyOrientation, options: [:])
        
        visionQueue.async {
            do {
                try requestHandler.perform([request])
            } catch {
                print("Failed to perform face detection: \(error)")
            }
        }
    }
    
    private func handleFaceDetection(request: VNRequest, error: Error?, pixelBuffer: CVPixelBuffer) {
        if let error = error {
            print("Face detection error: \(error)")
            return
        }

        guard let results = request.results as? [VNFaceObservation], !results.isEmpty else {
            DispatchQueue.main.async {
                self.emotion = "얼굴을 찾을 수 없음"
                print("No faces detected.")
            }
            return
        }

        for (index, face) in results.enumerated() {
            let boundingBox = face.boundingBox

            let imageWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let imageHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

            let adjustedX = boundingBox.origin.x * imageWidth
            let adjustedY = (1 - boundingBox.origin.y - boundingBox.height) * imageHeight
            let adjustedWidth = boundingBox.size.width * imageWidth
            let adjustedHeight = boundingBox.size.height * imageHeight

            let faceRect = CGRect(x: adjustedX, y: adjustedY, width: adjustedWidth, height: adjustedHeight)
            print("Final Bounding Box after adjustment: \(faceRect)")

            let croppedCIImage = CIImage(cvPixelBuffer: pixelBuffer).cropped(to: faceRect)

            // 흑백 필터 적용
            let grayscaleFilter = CIFilter.colorControls()
            grayscaleFilter.inputImage = croppedCIImage
            grayscaleFilter.saturation = 0.0 // 채도를 0으로 설정하여 흑백 변환
            grayscaleFilter.contrast = 1.1 // 대비를 조절하여 좀 더 명확하게 만듦
            
            guard let grayscaleImage = grayscaleFilter.outputImage else {
                print("Failed to apply grayscale filter to image.")
                return
            }

            // 이미지 방향 수정
            let rotatedImage = grayscaleImage.transformed(by: CGAffineTransform(rotationAngle: -.pi / 2))

            // 회전된 이미지 저장
            saveCIImageToPhotoLibrary(rotatedImage, forFace: index + 1)
            
            // 모델에 사용될 크기의 PixelBuffer로 변환
            guard let resizedPixelBuffer = resizeImageToPixelBuffer(ciImage: rotatedImage, targetSize: CGSize(width: 224, height: 224)) else {
                print("Failed to resize face image.")
                return
            }

            performEmotionPrediction(with: resizedPixelBuffer, forFace: index + 1)
        }
    }

    private func saveCIImageToPhotoLibrary(_ ciImage: CIImage, forFace faceNumber: Int) {
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            print("Failed to create CGImage from CIImage.")
            return
        }

        let uiImage = UIImage(cgImage: cgImage)
        UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil)
        print("Saved input image for face \(faceNumber) to Photo Library (for verification).")
    }

    private func resizeImageToPixelBuffer(ciImage: CIImage, targetSize: CGSize) -> CVPixelBuffer? {
        let attributes = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(targetSize.width),
            Int(targetSize.height),
            kCVPixelFormatType_32BGRA,
            attributes,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            print("Failed to create pixel buffer.")
            return nil
        }

        let ciContext = CIContext()
        ciContext.render(ciImage, to: buffer)

        return buffer
    }

    private func performEmotionPrediction(with pixelBuffer: CVPixelBuffer, forFace faceNumber: Int) {
        do {
            let input = ViT_EmotionDetection_ConvertedInput(x_1: pixelBuffer)
            let output = try emotionModel.prediction(input: input)
            let scores = output.linear_72

            let emotionsList = ["분노", "혐오", "공포", "행복", "슬픔", "놀람", "중립"]
            
            print("Emotion Scores for Face \(faceNumber):")
            for (index, emotion) in emotionsList.enumerated() {
                let score = scores[index].floatValue
                print("- \(emotion): \(score)")
            }

            let maxIndex = findArgmax(scores)

            if maxIndex < emotionsList.count && maxIndex >= 0 {
                DispatchQueue.main.async {
                    self.emotion = emotionsList[maxIndex]
                    print("Face \(faceNumber): Detected emotion: \(emotionsList[maxIndex]) with index \(maxIndex) and score \(scores[maxIndex])")
                }
            } else {
                DispatchQueue.main.async {
                    self.emotion = "알 수 없음"
                    print("Face \(faceNumber): Invalid emotion index: \(maxIndex)")
                }
            }
        } catch {
            print("모델 예측 오류: \(error)")
            DispatchQueue.main.async {
                self.emotion = "오류 발생"
            }
        }
    }

    private func findArgmax(_ multiArray: MLMultiArray) -> Int {
        var maxValue: Float = -Float.greatestFiniteMagnitude
        var maxIndex: Int = -1

        for i in 0..<multiArray.count {
            let value = multiArray[i].floatValue
            if value > maxValue {
                maxValue = value
                maxIndex = i
            }
        }
        return maxIndex
    }
}
