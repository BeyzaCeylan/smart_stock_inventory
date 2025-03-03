import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:typed_data';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Inventory Counting with Computer Vision',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late Interpreter _interpreter;
  File? _image;
  final picker = ImagePicker();
  List<Map<String, dynamic>> detectedObjects = [];

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  // Modeli yükle
  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/1.tflite');
      print("Model loaded successfully");

      // Modelin giriş ve çıkış tensörlerini kontrol et
      var inputTensors = _interpreter.getInputTensors();
      var outputTensors = _interpreter.getOutputTensors();

      print("Input Tensors: $inputTensors");
      print("Output Tensors: $outputTensors");
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  // Kameradan veya galeriden resim seçme
  Future<void> _getImage(ImageSource source) async {
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      await runInference();
    }
  }

  // Modeli çalıştırma ve çıktıları alabilme
  Future<void> runInference() async {
    // Null kontrolü
    if (_interpreter == null) {
      print("Interpreter not initialized yet.");
      return;
    }

    // Resmi yükle ve işleme
    var input = await _image?.readAsBytes();
    if (input == null) {
      print("Image not loaded.");
      return;
    }

    var image = img.decodeImage(input);
    if (image == null) {
      print("Image decoding failed.");
      return;
    }

    // Resmi 300x300 boyutuna getir
    image = img.copyResize(image, width: 300, height: 300);

    // RGB formatında dönüştürme
    img.Image rgbImage = img.Image.from(image);

    // UINT8 formatına dönüştürme
    List<int> uint8List = rgbImage.getBytes();
    var inputData = Uint8List.fromList(uint8List);

    // Quantization parametrelerini uygula
    inputData = Uint8List.fromList(inputData.map((value) => (value - 128) ~/ 1).toList());

    if (inputData == null) {
      print("Input data is null.");
      return;
    }

    // Çıktı için listeler oluştur
    var outputLocations = List.generate(1, (i) {
      return List.generate(10, (j) {
        return List.filled(4, 0.0);
      });
    });
    var outputClasses = List.generate(1, (i) => List.filled(10, 0.0));
    var outputScores = List.generate(1, (i) => List.filled(10, 0.0));
    var numDetections = List.filled(1, 0.0);

    // Modeli çalıştır
    try {
      _interpreter.runForMultipleInputs(
        [inputData.buffer.asUint8List()],
        {
          0: outputLocations,
          1: outputClasses,
          2: outputScores,
          3: numDetections,
        },
      );

      // Çıktıları işle
      processOutput(outputLocations, outputClasses, outputScores, numDetections);
    } catch (e) {
      print("Error running inference: $e");
    }
  }

  // Modelin çıktısını işleme
  void processOutput(
    List<List<List<double>>> outputLocations,
    List<List<double>> outputClasses,
    List<List<double>> outputScores,
    List<double> numDetections,
  ) {
    detectedObjects.clear();

    int numResults = numDetections[0].toInt();
    for (int i = 0; i < numResults; i++) {
      double score = outputScores[0][i];
      if (score > 0.5) {  // Güven skoru 0.5'ten büyükse
        List<double> boundingBox = outputLocations[0][i];
        int classId = outputClasses[0][i].toInt();

        // Sınıf ID'sine göre etiketi belirle
        String label = _getLabelFromClassId(classId);

        detectedObjects.add({
          'label': label,  // Etiket
          'confidence': score,  // Güven skoru
          'boundingBox': boundingBox,  // Bounding box koordinatları
        });
      }
    }

    setState(() {});
  }

  // Sınıf ID'sine göre etiketi belirle
  String _getLabelFromClassId(int classId) {
    switch (classId) {
      case 0:
        return 'Person';
      case 1:
        return 'Car';
      case 2:
        return 'Dog';
      case 3:
        return 'Cat';
      default:
        return 'Object $classId';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Inventory Counting with CV"),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image == null
                ? Text("No image selected.")
                : Stack(
                    children: [
                      Image.file(_image!),
                      ..._getBoundingBoxes(),
                    ],
                  ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => _getImage(ImageSource.camera),
              child: Text("Capture Image"),  // Kameradan resim çekme
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => _getImage(ImageSource.gallery),
              child: Text("Select Image from Gallery"),  // Galeriden resim seçme
            ),
          ],
        ),
      ),
    );
  }

  // Bounding box'ları çizme
  List<Widget> _getBoundingBoxes() {
    List<Widget> boxes = [];
    for (var detectedObject in detectedObjects) {
      var boundingBox = detectedObject['boundingBox'];
      var left = boundingBox[1] * 300;  // Xmin
      var top = boundingBox[0] * 300;   // Ymin
      var right = boundingBox[3] * 300; // Xmax
      var bottom = boundingBox[2] * 300;// Ymax

      boxes.add(Positioned(
        left: left,
        top: top,
        child: Container(
          width: right - left,
          height: bottom - top,
          decoration: BoxDecoration(
            border: Border.all(color: Colors.yellow, width: 3),
            borderRadius: BorderRadius.circular(5),
          ),
          child: Center(
            child: Text(
              detectedObject['label']!,
              style: TextStyle(color: Colors.yellow, fontSize: 16),
            ),
          ),
        ),
      ));
    }
    return boxes;
  }
}