import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;

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
  List<String> labels = [];

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  /// **📌 Modeli ve Etiketleri Yükle**
  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/1.tflite');
      print("✅ Model başarıyla yüklendi");

      // Modelin giriş ve çıkış tensörlerini kontrol et
      var inputTensors = _interpreter.getInputTensors();
      var outputTensors = _interpreter.getOutputTensors();

      print("📌 Giriş Tensors: $inputTensors");
      print("📌 Çıkış Tensors: $outputTensors");

      // Etiketleri oku
      await loadLabels();
    } catch (e) {
      print("🚨 Model yükleme hatası: $e");
    }
  }

  /// **📌 `labelmap.txt` dosyasını oku**
  Future<void> loadLabels() async {
    try {
      String labelsData = await rootBundle.loadString('assets/labelmap.txt');
      labels = labelsData.split('\n').where((label) => label.isNotEmpty).toList();
      print("✅ Etiketler yüklendi: $labels");
    } catch (e) {
      print("🚨 Etiket dosyası yüklenemedi: $e");
    }
  }

  /// **📌 Kameradan veya galeriden resim seç**
  Future<void> _getImage(ImageSource source) async {
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      await runInference();
    }
  }

  /// **📌 Modeli çalıştır ve nesne tespiti yap**
 Future<void> runInference() async {
  if (_interpreter == null) {
    print("🚨 Model yüklenmemiş!");
    return;
  }

  if (_image == null) {
    print("🚨 Resim seçilmedi!");
    return;
  }

  var input = await _image?.readAsBytes();
  if (input == null) {
    print("🚨 Resim yüklenemedi!");
    return;
  }

  var image = img.decodeImage(input);
  if (image == null) {
    print("🚨 Resim decode edilemedi!");
    return;
  }

  // **📌 Resmi 300x300 boyutuna getir**
  image = img.copyResize(image, width: 300, height: 300);

  // **📌 Giriş verisini `uint8` formatına çevir**
  Uint8List inputData = Uint8List(300 * 300 * 3);
  int pixelIndex = 0;

  for (int y = 0; y < 300; y++) {
    for (int x = 0; x < 300; x++) {
      img.Pixel pixel = image.getPixel(x, y);

      inputData[pixelIndex++] = pixel.r.toInt(); // Kırmızı bileşen
      inputData[pixelIndex++] = pixel.g.toInt(); // Yeşil bileşen
      inputData[pixelIndex++] = pixel.b.toInt(); // Mavi bileşen
    }
  }

  // **📌 Model çıktıları için tensörleri hazırla**
  var outputLocations = List.generate(1, (i) {
    return List.generate(10, (j) {
      return List.filled(4, 0.0);
    });
  });
  var outputClasses = List.generate(1, (i) => List.filled(10, 0.0));
  var outputScores = List.generate(1, (i) => List.filled(10, 0.0));
  var numDetections = List.filled(1, 0.0);

  // **📌 Modeli çalıştır**
  try {
    _interpreter.runForMultipleInputs(
      [inputData], // **Giriş verisi `uint8` olarak gönderiliyor**
      {
        0: outputLocations,
        1: outputClasses,
        2: outputScores,
        3: numDetections,
      },
    );

    processOutput(outputLocations, outputClasses, outputScores, numDetections);
  } catch (e) {
    print("🚨 Model çalıştırma hatası: $e");
  }
}



  /// **📌 Modelin çıktısını işle ve nesneleri belirle**
  void processOutput(
    List<List<List<double>>> outputLocations,
    List<List<double>> outputClasses,
    List<List<double>> outputScores,
    List<double> numDetections) {
  
  detectedObjects.clear(); // Önceki tespitleri temizle

  int numResults = numDetections[0].toInt(); // Algılanan nesne sayısını al
  for (int i = 0; i < numResults; i++) {
    double score = outputScores[0][i]; // Modelin güven skoru

    if (score > 0.6) {  // Eğer güven skoru 0.5’ten büyükse geçerli kabul et
      List<double> boundingBox = outputLocations[0][i]; // Nesnenin koordinatları
      int classId = outputClasses[0][i].toInt(); // Sınıf ID’sini al
      String detectedLabel = _getLabelFromClassId(classId); // ID'yi etikete çevir

      // Algılanan nesneyi listeye ekle
      detectedObjects.add({
        'label': detectedLabel,  // Örneğin: 'cat', 'banana' vs.
        'confidence': score,  // Güven skoru
        'boundingBox': boundingBox,  // Nesnenin koordinatları
      });
    }
  }

  setState(() {}); // Ekranı güncelle
}


  /// **📌 Sınıf ID'sine göre etiketi al**
  String _getLabelFromClassId(int classId) {
    if (classId < labels.length) {
      return labels[classId];
    } else {
      return 'Unknown';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Inventory Counting with CV")),
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
            ElevatedButton(onPressed: () => _getImage(ImageSource.camera), child: Text("Capture Image")),
            SizedBox(height: 20),
            ElevatedButton(onPressed: () => _getImage(ImageSource.gallery), child: Text("Select Image from Gallery")),
          ],
        ),
      ),
    );
  }

  /// **📌 Bounding box'ları çiz**
  List<Widget> _getBoundingBoxes() {
    List<Widget> boxes = [];
    for (var detectedObject in detectedObjects) {
      var boundingBox = detectedObject['boundingBox'];
      boxes.add(Positioned(
        left: boundingBox[1] * 300,
        top: boundingBox[0] * 300,
        child: Container(
          width: boundingBox[3] * 300 - boundingBox[1] * 300,
          height: boundingBox[2] * 300 - boundingBox[0] * 300,
          decoration: BoxDecoration(border: Border.all(color: Colors.yellow, width: 3)),
          child: Center(child: Text(detectedObject['label'], style: TextStyle(color: Colors.yellow))),
        ),
      ));
    }
    return boxes;
  }
}
