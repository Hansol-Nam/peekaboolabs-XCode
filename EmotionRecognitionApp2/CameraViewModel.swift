import SwiftUI
import AVFoundation
import Vision
import CoreML
import CoreImage
import CoreImage.CIFilterBuiltins
import UIKit
import os.log // 로그 기록을 위한 import
import Foundation

extension Array where Element: Comparable {
    func argmax() -> Int? {
        guard let maxValue = self.max() else { return nil }
        return self.firstIndex(of: maxValue)
    }
}


class CameraViewModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var emotion: String = "알 수 없음"
    
    public let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    
    private let emotionModel: coreml_model_test_float16
    private let context = CIContext() // CIContext를 클래스 수준에서 생성해 매번 재사용
    
    private let visionQueue = DispatchQueue(label: "vision.queue")
    
    // 디버그 모드 플래그
    private let debugMode = false
    private var frameCount = 0 // 프레임 카운터

    override init() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine // GPU & Neural Engine을 최대한 활용
            self.emotionModel = try coreml_model_test_float16(configuration: config)
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
            self.session.sessionPreset = .high // 처리량 감소를 위해 해상도 낮춤

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
        
        // 프레임 드롭: 초당 30fps 기준으로 5fps로 줄이기
        frameCount += 1
        if frameCount % 6 != 0 { // 홀수 프레임은 건너뜀
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
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let boundingBox = face.boundingBox

            let imageWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let imageHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

            let adjustedX = boundingBox.origin.x * imageWidth
            let adjustedY = (1 - boundingBox.origin.y - boundingBox.height) * imageHeight
            let adjustedWidth = boundingBox.size.width * imageWidth
            let adjustedHeight = boundingBox.size.height * imageHeight

            var faceRect = CGRect(x: adjustedX, y: adjustedY, width: adjustedWidth, height: adjustedHeight)

            // faceRect가 이미지의 extent를 벗어나지 않도록 보정
            faceRect = faceRect.intersection(ciImage.extent)

            if faceRect.isEmpty {
                print("Face bounding box does not intersect with the image extent.")
                continue
            }
            if faceRect.width <= 0 || faceRect.height <= 0 {
                print("Invalid faceRect dimensions: width = \(faceRect.width), height = \(faceRect.height)")
                continue
            }

            print("Final Bounding Box after adjustment: \(faceRect)")

            var croppedCIImage = ciImage.cropped(to: faceRect)

            if croppedCIImage.extent.width <= 0 || croppedCIImage.extent.height <= 0 {
                print("Invalid cropped CIImage extent: \(croppedCIImage.extent)")
                continue
            }
            
            // 잘라낸 이미지의 원점을 (0, 0)으로 이동하여 크기 보정
            croppedCIImage = croppedCIImage.transformed(by: CGAffineTransform(translationX: -croppedCIImage.extent.origin.x,
                                                                              y: -croppedCIImage.extent.origin.y))
            print("Corrected Cropped CIImage extent: \(croppedCIImage.extent)")

            // 흑백 필터 적용
            let grayscaleFilter = CIFilter.colorControls()
            grayscaleFilter.inputImage = croppedCIImage
            grayscaleFilter.saturation = 0.0 // 채도를 0으로 설정하여 흑백 변환
            grayscaleFilter.contrast = 1.1 // 대비를 조절하여 좀 더 명확하게 만듦

            guard let grayscaleImage = grayscaleFilter.outputImage else {
                print("Failed to apply grayscale filter to image.")
                continue
            }

            // 이미지 방향 수정
            let rotatedImage = grayscaleImage.transformed(by: CGAffineTransform(rotationAngle: -.pi / 2))

            // 회전된 이미지의 크기가 유효한지 검증
            if rotatedImage.extent.width <= 0 || rotatedImage.extent.height <= 0 || rotatedImage.extent.origin.x.isInfinite || rotatedImage.extent.origin.y.isInfinite {
                print("Rotated image extent is invalid.")
                continue
            }

            saveCIImageToPhotoLibrary(rotatedImage, forFace: index + 1)

            // 모델에 사용될 크기의 PixelBuffer로 변환
            guard let resizedPixelBuffer = resizeImageToPixelBuffer(ciImage: rotatedImage, targetSize: CGSize(width: 224, height: 224)) else {
                print("Failed to resize face image.")
                continue
            }

            performEmotionPrediction(with: resizedPixelBuffer, forFace: index + 1)
        }
    }



    private func saveCIImageToPhotoLibrary(_ ciImage: CIImage, forFace faceNumber: Int) {
        
        guard debugMode else {
                print("Debug mode is off. Skipping image saving.")
                return
            }
        
        
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

        // CIImage를 원점(0,0)에서 시작하도록 보정하여 음수 좌표 제거
        var normalizedImage = ciImage
        if normalizedImage.extent.origin.x != 0 || normalizedImage.extent.origin.y != 0 {
            let translationTransform = CGAffineTransform(translationX: -normalizedImage.extent.origin.x,
                                                         y: -normalizedImage.extent.origin.y)
            normalizedImage = normalizedImage.transformed(by: translationTransform)
        }

        print("Normalized CIImage extent: \(normalizedImage.extent)")
        print("Target size for resizing: \(targetSize)")

        // CIContext 렌더링을 위한 보정
        let ciContext = CIContext()

        // 이미지를 targetSize로 스케일링
        let scaleX = targetSize.width / normalizedImage.extent.width
        let scaleY = targetSize.height / normalizedImage.extent.height

        // 스케일 값이 유효한지 확인합니다.
        if scaleX.isInfinite || scaleY.isInfinite || scaleX <= 0 || scaleY <= 0 {
            print("Invalid scale values: scaleX = \(scaleX), scaleY = \(scaleY)")
            return nil
        }

        let scaledImage = normalizedImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        print("Scaled CIImage extent: \(scaledImage.extent)")

        // 렌더링 대상과 이미지 크기 일치
        let targetRect = CGRect(origin: .zero, size: targetSize)
        ciContext.render(scaledImage, to: buffer, bounds: targetRect, colorSpace: CGColorSpaceCreateDeviceRGB())

        return buffer
    }


    func createMLMultiArray(from pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        let shape: [NSNumber] = [1, 3, 224, 224] // 1(batch) x 3(channels) x 224(height) x 224(width)
        guard let mlMultiArray = try? MLMultiArray(shape: shape, dataType: .float32) else {
            print("Failed to create MLMultiArray.")
            return nil
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("Pixel buffer base address is nil.")
            return nil
        }

        let height = CVPixelBufferGetHeight(pixelBuffer)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let channels = 3

        let pointer = baseAddress.assumingMemoryBound(to: UInt8.self)
        var index = 0
        for c in 0..<channels {
            for h in 0..<height {
                for w in 0..<width {
                    let pixelIndex = h * bytesPerRow + w * channels + c
                    mlMultiArray[index] = NSNumber(value: Float(pointer[pixelIndex]) / 255.0) // Normalize [0, 1]
                    index += 1
                }
            }
        }
        return mlMultiArray
    }
    
    private func performEmotionPrediction(with pixelBuffer: CVPixelBuffer, forFace faceNumber: Int) {
        // Core ML 모델 실행을 별도의 글로벌 큐에서 수행
        DispatchQueue.global(qos: .userInitiated).async {[weak self] in
            guard let self = self else { return }
            // 추론 시작 시각 기록
            //let startTime = CFAbsoluteTimeGetCurrent()

            // 메모리 사용량 측정 (시작 전)
            //let memoryUsageStart = reportMemory()
            
            // PixelBuffer 정보 확인 로그
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
            
            print("PixelBuffer Information:")
            print("- Width: \(width)")
            print("- Height: \(height)")
            print("- Pixel Format: \(pixelFormat == kCVPixelFormatType_32BGRA ? "32BGRA" : "Unknown")")

            // 픽셀 버퍼의 내용 일부를 출력하여 확인 (중간 값 일부를 체크)
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            if let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) {
                let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
                let samplePixel = (width * height) / 2 // 중간 픽셀의 위치
                print("Sample Pixel Values (RGBA): R: \(buffer[samplePixel * 4]), G: \(buffer[samplePixel * 4 + 1]), B: \(buffer[samplePixel * 4 + 2]), A: \(buffer[samplePixel * 4 + 3])")
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            
            guard let mlMultiArray = self.createMLMultiArray(from: pixelBuffer) else {
                    print("Failed to create MLMultiArray from pixel buffer.")
                    return
                }
            
            do {
                let input = coreml_model_test_float16Input(x_1: mlMultiArray)
                let output = try self.emotionModel.prediction(input: input)
                let scoresArray = output.linear_72ShapedArray.scalars // [Float] 배열로 변환

                //let emotionsList = ["분노", "혐오", "공포", "행복", "슬픔", "놀람", "중립"]
                let emotionsList = ["sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"]

                print("Emotion Scores for Face \(faceNumber):")
                
                for (index, emotion) in emotionsList.enumerated() {
                    print("- \(emotion): \(scoresArray[index])")
                }

                // 최고 점수의 감정 확인
                if let maxIndex = scoresArray.argmax() {
                    print("Detected Emotion: \(emotionsList[maxIndex])")
                    if self.emotion != emotionsList[maxIndex] {
                        DispatchQueue.main.async {
                            self.emotion = emotionsList[maxIndex]
                        }
                    }
                } else {
                    DispatchQueue.main.async {
                        self.emotion = "알 수 없음"
                        print("Face \(faceNumber): Invalid emotion index.")
                    }
                }
                
                
                // 추론 종료 시각 기록
                //let endTime = CFAbsoluteTimeGetCurrent()
                //let elapsedTime = endTime - startTime
                //print("Inference Time for Face \(faceNumber): \(elapsedTime) seconds")

                // 메모리 사용량 측정 (추론 후)
                //let memoryUsageEnd = reportMemory()
                //let memoryUsageDifference = memoryUsageEnd - memoryUsageStart
                //print("Memory Usage Difference for Face \(faceNumber): \(memoryUsageDifference / 1024) KB")
                
                
            } catch {
                print("모델 예측 오류: \(error)")
                DispatchQueue.main.async {
                    self.emotion = "오류 발생"
                }
            }
        }
    }
    
    
    // 메모리 사용량을 반환하는 함수
    private func reportMemory() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size) / 4

        let result: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }

        if result == KERN_SUCCESS {
            return info.phys_footprint
        } else {
            return 0 // 메모리 사용량을 가져오지 못한 경우
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
