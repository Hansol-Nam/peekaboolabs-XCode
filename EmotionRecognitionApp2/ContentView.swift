import SwiftUI
import AVFoundation
import Vision
import CoreML

struct ContentView: View {
    @ObservedObject var cameraViewModel = CameraViewModel()

    var body: some View {
        VStack {
            CameraPreview(session: cameraViewModel.session)
                .edgesIgnoringSafeArea(.all)
                .onAppear {
                    cameraViewModel.checkPermissions()
                    cameraViewModel.configureSession()
                }
            
            Text(cameraViewModel.emotion)
                .font(.largeTitle)
                .padding()
        }
    }
}
