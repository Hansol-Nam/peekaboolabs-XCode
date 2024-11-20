//
//  EmotionRecognitionApp2App.swift
//  EmotionRecognitionApp2
//
//  Created by hansol on 11/19/24.
//

import SwiftUI

@main
struct EmotionRecognitionApp2App: App {
    init() {
        // Metal 지원 여부 확인
        if MTLCreateSystemDefaultDevice() == nil {
            print("Metal is not supported on this device.")
        } else {
            print("Metal is supported.")
        }
    }
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
