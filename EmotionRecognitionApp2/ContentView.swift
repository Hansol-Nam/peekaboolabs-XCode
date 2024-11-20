// ContentView.swift

import SwiftUI

struct ContentView: View {
    @StateObject private var cameraViewModel = CameraViewModel()
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                CameraPreview(session: cameraViewModel.session)
                    .ignoresSafeArea()
                
                // Bounding Box 그리기
                ForEach(cameraViewModel.boundingBoxes) { box in
                    BoundingBoxView(rect: box.rect, imageSize: getImageSize(), viewSize: geometry.size)
                }
                
                VStack {
                    Spacer()
                    
                    // 감정 표시
                    Text("감정: \(cameraViewModel.emotions.first ?? "알 수 없음")")
                        .font(.largeTitle)
                        .padding()
                        .background(Color.black.opacity(0.5))
                        .foregroundColor(.white)
                        .cornerRadius(10)
                        .padding()
                    
                    // 크롭된 얼굴 이미지 표시 (디버깅용)
                    ScrollView(.horizontal, showsIndicators: true) {
                        HStack {
                            // ForEach를 수정하여 인덱스를 id로 사용
                            ForEach(Array(cameraViewModel.croppedFaceImages.enumerated()), id: \.offset) { index, image in
                                Image(uiImage: image)
                                    .resizable()
                                    .frame(width: 100, height: 100)
                                    .border(Color.blue, width: 2)
                            }
                        }
                        .padding()
                    }
                }
            }
        }
    }
    
    // 실제 카메라 이미지의 크기를 반환하는 함수
    private func getImageSize() -> CGSize {
        // 실제 카메라 이미지의 해상도에 맞게 설정
        // 로그에 따르면, 현재 이미지 크기는 1920x1080 (width x height)입니다.
        return CGSize(width: 1920, height: 1080)
    }
}

struct BoundingBoxView: View {
    var rect: CGRect
    var imageSize: CGSize
    var viewSize: CGSize
    
    var body: some View {
        GeometryReader { geometry in
            let scaleX = geometry.size.width / imageSize.width
            let scaleY = geometry.size.height / imageSize.height
            
            Rectangle()
                .stroke(Color.red, lineWidth: 2)
                .frame(width: rect.width * scaleX, height: rect.height * scaleY)
                .position(x: (rect.midX * scaleX), y: (rect.midY * scaleY))
        }
        .allowsHitTesting(false)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
