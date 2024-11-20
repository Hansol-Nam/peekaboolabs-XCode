// CameraViewModel.swift

import SwiftUI
import AVFoundation
import Vision
import CoreML

struct FaceBoundingBox: Identifiable {
    let id = UUID()
    let rect: CGRect
}

class CameraViewModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    public let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    
    @Published var emotion: String = "알 수 없음"
    @Published var boundingBoxes: [FaceBoundingBox] = []
    @Published var emotions: [String] = [] // 각 얼굴에 대한 감정을 저장할 배열 추가
    @Published var croppedFaceImages: [UIImage] = [] // 크롭된 얼굴 이미지를 저장할 배열 추가
    
    private let emotionModel: ViT_EmotionDetection_Converted
    
    // Vision Request Handler를 위한 큐
    private let visionQueue = DispatchQueue(label: "vision.queue")
    
    override init() {
        // 모델 초기화
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly // Metal 관련 문제를 우회하기 위해 CPU 전용 사용
            self.emotionModel = try ViT_EmotionDetection_Converted(configuration: config)
        } catch {
            fatalError("모델을 로드할 수 없습니다: \(error)")
        }
        
        super.init()
        setupVision()
        checkPermissions()
        configureSession()
    }
    
    // Vision 얼굴 감지 요청 설정
    private func setupVision() {
        // 별도의 VNDetectFaceRectanglesRequest를 설정할 필요 없이, 요청을 직접 생성하여 사용
    }
    
    func checkPermissions() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            // 이미 권한이 있음
            break
        case .notDetermined:
            // 권한 요청
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if !granted {
                    print("카메라 접근 권한이 필요합니다.")
                }
            }
        default:
            // 권한이 없음
            print("카메라 접근 권한이 필요합니다.")
        }
    }
    
    func configureSession() {
        sessionQueue.async {
            self.session.beginConfiguration()
            
            // 해상도 설정
            self.session.sessionPreset = .high
            
            // 후면 카메라 입력 설정
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
            
            // 비디오 출력 설정
            let videoOutput = AVCaptureVideoDataOutput()
            // 모델의 입력 요구사항에 맞게 픽셀 포맷 설정
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "video.output.queue"))
            videoOutput.alwaysDiscardsLateVideoFrames = true
            if self.session.canAddOutput(videoOutput) {
                self.session.addOutput(videoOutput)
            }
            
            self.session.commitConfiguration()
            self.session.startRunning()
            
            // 로그: 세션 설정 완료
            print("Camera session configured and started.")
        }
    }
    
    // AVCaptureVideoDataOutputSampleBufferDelegate 메서드
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // 이미지 처리
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to get pixel buffer from sample buffer.")
            return
        }
        detectFace(on: pixelBuffer)
    }
    
    // 얼굴 감지 및 감정 인식 함수
    func detectFace(on pixelBuffer: CVPixelBuffer) {
        let deviceOrientation = UIDevice.current.orientation
        var cgImagePropertyOrientation: CGImagePropertyOrientation
        
        switch deviceOrientation {
        case .portrait:
            cgImagePropertyOrientation = .up
        case .portraitUpsideDown:
            cgImagePropertyOrientation = .down
        case .landscapeLeft:
            cgImagePropertyOrientation = .left
        case .landscapeRight:
            cgImagePropertyOrientation = .right
        default:
            cgImagePropertyOrientation = .up // 기본 방향으로 설정
        }
        
        // 로그: 이미지 방향 확인
        print("Device orientation: \(deviceOrientation.rawValue), CGImagePropertyOrientation: \(cgImagePropertyOrientation.rawValue)")
        
        // 로그: 이미지 크기 확인
        let imageWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let imageHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        print("Processing image of size: \(imageWidth) x \(imageHeight)")
        
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
    
    // 얼굴 감지 결과 처리
    private func handleFaceDetection(request: VNRequest, error: Error?, pixelBuffer: CVPixelBuffer) {
        if let error = error {
            print("Face detection error: \(error)")
            return
        }
        
        guard let results = request.results as? [VNFaceObservation], !results.isEmpty else {
            DispatchQueue.main.async {
                self.emotion = "얼굴을 찾을 수 없음"
                self.boundingBoxes = []
                self.emotions = []
                self.croppedFaceImages = []
                print("No faces detected.")
            }
            return
        }
        
        var newBoundingBoxes: [FaceBoundingBox] = []
        var newCroppedFaceImages: [UIImage] = []
        var newEmotions: [String] = []
        
        for (index, face) in results.enumerated() {
            let boundingBox = face.boundingBox
            
            // Vision의 boundingBox는 [0,1] 비율이므로 실제 이미지 크기로 변환
            let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            
            let faceRect = CGRect(x: boundingBox.origin.x * width,
                                  y: (1 - boundingBox.origin.y - boundingBox.size.height) * height,
                                  width: boundingBox.size.width * width,
                                  height: boundingBox.size.height * height)
            
            newBoundingBoxes.append(FaceBoundingBox(rect: faceRect))
            
            // 로그: Bounding Box 좌표 및 크기
            print("Face \(index + 1): CGRect(x: \(faceRect.origin.x), y: \(faceRect.origin.y), width: \(faceRect.size.width), height: \(faceRect.size.height))")
            
            // 얼굴 이미지를 크롭하고 리사이즈
            guard let croppedImage = cropPixelBuffer(pixelBuffer: pixelBuffer, to: faceRect) else {
                print("Failed to crop face image for face \(index + 1).")
                continue
            }
            print("Cropped image size: \(croppedImage.width) x \(croppedImage.height)")
            
            guard let resizedPixelBuffer = resizePixelBuffer(croppedImage, size: CGSize(width: 224, height: 224)) else {
                print("Failed to resize face image for face \(index + 1).")
                continue
            }
            print("Resized image size: 224 x 224")
            
            // UIImage로 변환하여 저장 (디버깅용)
            let ciImage = CIImage(cvPixelBuffer: resizedPixelBuffer)
            let uiImage = UIImage(ciImage: ciImage)
            
            // 모델에 예측 요청
            performEmotionPrediction(with: resizedPixelBuffer, forFace: index + 1)
        }
        
        DispatchQueue.main.async {
            self.boundingBoxes = newBoundingBoxes
            self.croppedFaceImages = newCroppedFaceImages
            self.emotions = newEmotions
        }
    }
    
    // PixelBuffer에서 얼굴 영역을 크롭하는 함수
    private func cropPixelBuffer(pixelBuffer: CVPixelBuffer, to rect: CGRect) -> CGImage? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        guard let context = CGContext(data: baseAddress,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else { return nil }
        
        guard let cgImage = context.makeImage() else { return nil }
        let croppedCGImage = cgImage.cropping(to: rect)
        return croppedCGImage
    }
    
    // 크롭된 CGImage를 224x224로 리사이즈하고 CVPixelBuffer로 변환
    private func resizePixelBuffer(_ cgImage: CGImage, size: CGSize) -> CVPixelBuffer? {
        let resizedImage = resizeImage(image: cgImage, targetSize: size)
        if let buffer = resizedImage.toCVPixelBuffer() {
            print("Successfully resized and converted image to CVPixelBuffer.")
            return buffer
        } else {
            print("Failed to convert resized image to CVPixelBuffer.")
            return nil
        }
    }
    
    // 모델 예측 함수
    private func performEmotionPrediction(with pixelBuffer: CVPixelBuffer, forFace faceNumber: Int) {
        do {
            let input = ViT_EmotionDetection_ConvertedInput(x_1: pixelBuffer)
            let output = try emotionModel.prediction(input: input)
            let scores = output.linear_72 // MLMultiArray
            
            // argmax 함수 사용
            let maxIndex = findArgmax(scores)
            
            let emotionsList = ["분노", "혐오", "공포", "행복", "슬픔", "놀람", "중립"]
            if maxIndex < emotionsList.count && maxIndex >= 0 {
                DispatchQueue.main.async {
                    if self.emotions.count > faceNumber - 1 {
                        self.emotions[faceNumber - 1] = emotionsList[maxIndex]
                    } else {
                        self.emotions.append(emotionsList[maxIndex])
                    }
                    print("Face \(faceNumber): Detected emotion: \(emotionsList[maxIndex]) with index \(maxIndex)")
                }
            } else {
                DispatchQueue.main.async {
                    if self.emotions.count > faceNumber - 1 {
                        self.emotions[faceNumber - 1] = "알 수 없음"
                    } else {
                        self.emotions.append("알 수 없음")
                    }
                    print("Face \(faceNumber): Invalid emotion index: \(maxIndex)")
                }
            }
        } catch {
            print("모델 예측 오류: \(error)")
            DispatchQueue.main.async {
                if self.emotions.count > faceNumber - 1 {
                    self.emotions[faceNumber - 1] = "오류 발생"
                } else {
                    self.emotions.append("오류 발생")
                }
            }
        }
    }
    
    // argmax 함수 구현
    func findArgmax(_ multiArray: MLMultiArray) -> Int {
        var maxValue: Float = -Float.greatestFiniteMagnitude
        var maxIndex: Int = -1

        for i in 0..<multiArray.count {
            let value = multiArray[i].floatValue
            print("Score at index \(i): \(value)") // 각 인덱스의 값 출력
            if value > maxValue {
                maxValue = value
                maxIndex = i
            }
        }
        print("argmax: index=\(maxIndex), value=\(maxValue)")
        return maxIndex
    }
    
    // 이미지 리사이즈 함수
    func resizeImage(image: CGImage, targetSize: CGSize) -> CGImage {
        let width = Int(targetSize.width)
        let height = Int(targetSize.height)
        let bitsPerComponent = image.bitsPerComponent
        let colorSpace = image.colorSpace ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = image.bitmapInfo
        
        guard let context = CGContext(data: nil,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: bitsPerComponent,
                                      bytesPerRow: 0,
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo.rawValue) else {
            return image
        }
        
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(origin: .zero, size: targetSize))
        
        guard let resizedImage = context.makeImage() else {
            return image
        }
        
        return resizedImage
    }
}

// CGImagePropertyOrientation 초기화 확장
extension CGImagePropertyOrientation {
    init?(_ deviceOrientation: UIDeviceOrientation) {
        switch deviceOrientation {
        case .portrait:
            self = .up
        case .portraitUpsideDown:
            self = .down
        case .landscapeLeft:
            self = .left
        case .landscapeRight:
            self = .right
        default:
            self = .up
        }
    }
}

// CVPixelBuffer 변환을 위한 확장
extension CGImage {
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let width = self.width
        let height = self.height
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }
        
        context.draw(self, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return buffer
    }
}
