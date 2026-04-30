import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const PWrScannerApp());
}

class PWrScannerApp extends StatelessWidget {
  const PWrScannerApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    const Color col =  Color(0xFF9A342D);
    return MaterialApp(
      title: 'PWr Scanner',
      theme: ThemeData(
        primaryColor: col,
        appBarTheme: const AppBarTheme(backgroundColor: col,),
        //colorScheme: .fromSeed(seedColor: Colors.col),
      ),
      home: const MainScreen(),
    );
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  File? _image; // zrobione zdjecie
  String _buildingResult = "Wybierz/zrób zdjęcie, żeby rozpoznać budynek";
  bool _isLoading = false;
  final ImagePicker _picker = ImagePicker();

  // robienie zdjecia
  Future<void> _takePhoto() async{
    final XFile? pickedFile = await _picker.pickImage(source: ImageSource.camera);
    if (pickedFile!=null){
      setState(() {
        _image = File(pickedFile.path);
        _buildingResult = "Analizowanie...";
      });

      await _uploadImageToServer(_image!);
    }
  }

  // wybieranie z galerii
  Future<void> _pickFromGallery() async{
    final XFile? pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile!=null){
      setState(() {
        _image = File(pickedFile.path);
        _buildingResult = "Analizowanie...";
      });
      await _uploadImageToServer(_image!);
    }
  }

  // wysylanie na serwer
  Future<void> _uploadImageToServer(File imageFile) async{
    setState(() {
      _isLoading=true;
    });

    try{
      // ADRES SERWERA
      String serverIP = "172.20.10.4";
      var uri = Uri.parse('http://$serverIP:8000/predict');

      // paczka ze zdjeciem
      var request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));

      // wysylamy na serwer i czekamy na odp
      var response = await request.send();

      // jesli OK
      if (response.statusCode == 200){
        var responseData = await response.stream.bytesToString();
        var json = jsonDecode(responseData);

        setState(() {
_buildingResult = "Budynek: ${json['budynek']}\nPrawdopodobieństwo: ${(json['pewnosc']).toStringAsFixed(2)}%";        });
      } else {
        setState(() {
          _buildingResult = "Błąd serwera. Kod: ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        _buildingResult = "Błąd połączenia!\nSprawdź adres IP i sieć Wi-Fi.";
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

 

  @override
  Widget build(BuildContext context) {
    const Color cols =  Color(0xFF9A342D);
    return Scaffold(
      appBar: AppBar(
        backgroundColor: cols,
        //backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('PWr Building Scanner', style: TextStyle(color: Colors.white)),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              height: 300,
              width: 300,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(15),
              ),
              child: _image == null
                  ? const Center(child: Icon(Icons.image, size: 100, color: Colors.grey))
                  : ClipRRect(
                      borderRadius: BorderRadius.circular(15),
                      child: Image.file(_image!, fit: BoxFit.cover),
                  ),
            ),
            const SizedBox (height: 30),


            _isLoading
                ? const CircularProgressIndicator()
                : Text (
                    _buildingResult,
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
            
            const SizedBox(height: 40),

            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _takePhoto,
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Aparat'),
                  style: ElevatedButton.styleFrom( 
                    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                  ),
                ),

                ElevatedButton.icon(
                  onPressed: _pickFromGallery,
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Galeria'),
                  style: ElevatedButton.styleFrom( 
                    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}