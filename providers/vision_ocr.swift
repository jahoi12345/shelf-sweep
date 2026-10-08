// Local Apple Vision text-only worker. Native image bytes; no language correction,
// custom catalogue words or title/author/edition inference. JSONL stdin/stdout.
import Foundation
import Vision
import ImageIO

func emit(_ value: [String: Any]) {
    do {
        let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        print(String(data: data, encoding: .utf8)!)
        fflush(stdout)
    } catch { fputs("JSON output failure\n", stderr) }
}
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesCPUOnly = true
request.usesLanguageCorrection = false
request.automaticallyDetectsLanguage = false
request.minimumTextHeight = 0
let languages: [String]
do { languages = try request.supportedRecognitionLanguages() }
catch { emit(["error": String(describing: error)]); exit(1) }
let wanted = ["es-ES", "en-US"]
guard wanted.allSatisfy({ languages.contains($0) }) else {
    emit(["error": "Spanish and English OCR languages unavailable", "supported_languages": languages]); exit(1)
}
request.recognitionLanguages = wanted
emit(["ready": true, "engine": "Apple Vision VNRecognizeTextRequest", "revision": request.revision,
      "supported_languages": languages, "recognition_languages": wanted, "language_correction": false,
      "automatic_language_detection": false, "requested_cpu_only": true,
      "os_version": ProcessInfo.processInfo.operatingSystemVersionString])
while let line = readLine() {
    autoreleasepool {
        do {
            let input = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
            guard let path = input["image_path"] as? String else { throw NSError(domain: "OCR", code: 1) }
            let degrees = input["rotation_deg_clockwise"] as? Int ?? 0
            let orientations: [Int: CGImagePropertyOrientation] = [0: .up, 90: .right, 180: .down, 270: .left]
            guard let orientation = orientations[degrees] else { throw NSError(domain: "OCR", code: 2) }
            let start = ProcessInfo.processInfo.systemUptime
            let handler = VNImageRequestHandler(url: URL(fileURLWithPath: path), orientation: orientation, options: [:])
            try handler.perform([request])
            let rows: [[String: Any]] = (request.results ?? []).map { observation in
                let box = observation.boundingBox
                return ["bbox_normalized_bottom_left": [box.minX, box.minY, box.maxX, box.maxY],
                        "candidates": observation.topCandidates(3).map { ["text": $0.string, "confidence": $0.confidence] }]
            }
            emit(["lines": rows, "rotation_deg_clockwise": degrees,
                  "latency_s": ProcessInfo.processInfo.systemUptime-start])
        } catch { emit(["error": String(describing: error)]) }
    }
}
