# Laporan Proyek Machine Learning On Production - Izzan Silmi Aziz 

![foto iris](https://i.imgur.com/BUcKu6v.png)

## Business Understanding
Bunga iris, yang secara ilmiah dikenal sebagai Iris, merupakan genus tumbuhan berbunga yang khas. Dalam genus ini, terdapat tiga spesies utama: Iris setosa, Iris versicolor, 
dan Iris virginica. Spesies ini menunjukkan variasi ciri fisiknya, terutama pada ukuran panjang sepal, lebar sepal, panjang kelopak, dan lebar kelopak.

### Problem Statements
Berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
-  Bagaimana membuat model machine learning yang dapat mengklasifikasi spesies bungan Iris dengan bantuan data variasi ciri fisiknya?
-  Model yang seperti apa yang memiliki akurasi paling baik?

### Goals
Tujuan dari proyek ini adalah:
- Mengembangkan model pembelajaran mesin yang mampu belajar dari pengukuran bunga iris.
- Mengklasifikasikannya secara akurat ke dalam spesiesnya masing-masing. 
- Mengotomatiskan proses klasifikasi berdasarkan karakteristik berbeda dari setiap spesies iris.

## Data Understanding
### EDA - Deskripsi Variabel
**Informasi Datasets**

| Jenis | Keterangan |
| ------ | ------ |
| Title | Iris Species |
| Source | [Kaggle](https://www.kaggle.com/datasets/uciml/iris/data) | 
| License | CC0: Public Domain |
| Visibility | Publik |
| Tags | Biology |
| Usability | 7.94 |

Berikut informasi pada dataset: 
Data yang digunakan dalam pembuatan model merupakan data primer, data ini didapat dari sebuah Universitas di Amerika, 
yang disediakan secara publik di kaggle dengan nama datasets yaitu: Iris Species.

![Tabel Data](https://github.com/isa96/MLP/blob/main/assets/1.PNG "Tabel Data")

Tabel 1. EDA Deskripsi Variabel

Dilihat dari Tabel 1. dataset ini telah di sesuaikan terlebih dahulu oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula. 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 150 sample dengan 5 fitur.
- Dataset memiliki 4 fitur bertipe float64 dan 1 fitur bertipe object.

![Tabel Data Duplikat](https://github.com/isa96/MLP/blob/main/assets/2.PNG "Tabel Data Duplikat")

Tabel 2. Tabel Data Duplikasi

Ada data yang duplikat, oleh karena itu harus diperiksa apakah setiap kumpulan data di kolom class seimbang atau tidak.

![Data Series Species](https://github.com/isa96/MLP/blob/main/assets/3.PNG "Data Series Species")

Gambar 1. Jumlah Variasi Per Kelas Dari Kolom Class

Karena data seimbang maka jangan menghapus duplikasi data karena dapat menyebabkan ketidakseimbangan kumpulan data.
Setiap class (iris-sentosa, versicolor, virginica) memiliki jumlah 50 data.

### EDA - Bivariate Analysis

![Analisis Bivariat (Sepal)](https://github.com/isa96/MLP/blob/main/assets/4.PNG "Analisis Bivariat (Sepal)")

Gambar 2a. Analisis Bivariat (Data Sepal) 

![Analisis Bivariat (Petal)](https://github.com/isa96/MLP/blob/main/assets/5.PNG "Analisis Bivariat (Petal)")

Gambar 2b. Analisis Bivariat (Data Petal)

Di Gambar 2a dan 2b, informasi yang dapat diketahui yaitu:
  - Class Iris Setosa memiliki panjang sepal yang lebih kecil tetapi ukuran lebar sepal sangat tinggi.
  - Class Iris Versicolor terletak di tengah untuk panjang dan lebar.
  - Class Iris Virginica memiliki panjang sepal yang lebih besar dan lebar sepal yang lebih kecil.
  - Class Iris Setosa memiliki panjang kelopak(petal) dan lebar kelopak(petal) terkecil.
  - Class Iris Versicolor memiliki panjang kelopak(petal) dan lebar kelopak(petal) rata-rata.
  - Class Iris Virginica memiliki panjang kelopak dan lebar kelopak tertinggi.

 ### EDA - Multivariate Analysis

![Multivariate Analysis](https://github.com/isa96/MLP/blob/main/assets/6.PNG "Analisis Multivariat")


Gambar 3a. Analisis Multivariat

![Multivariate Analysis](https://github.com/isa96/MLP/blob/main/assets/7.PNG "Analisis Matriks Korelasi")


Gambar 3b. Analisis Matriks Korelasi

Pada Gambar 3a dan 3b, informasi yang dapat diketahui yaitu:
  - Class Iris Setosa memiliki panjang dan lebar kelopak(petal) yang rendah.
  - Class Iris Versicolor memiliki panjang dan lebar kelopak(petal) rata-rata.
  - Class Iris Virginica memiliki panjang dan lebar kelopak(petal) yang tinggi.
  - Class Iris Setosa memiliki lebar sepal tinggi dan panjang sepal yang rendah.
  - Class Iris Versicolor memiliki panjang dan lebar sepal rata-rata.
  - Class Iris Virginica memiliki lebar yang kecil tetapi panjang sepal yang besar.
  - Korelasi tinggi antara kolom panjang dan lebar kelopak(petal)
  - Panjang Sepal dan Lebar Sepal sedikit berkorelasi satu sama lain

### Splitting

Pada projek ini digunakan _Train Test Split_ pada library  *sklearn.model_selection* untuk membagi dataset menjadi data latih dan data uji dengan pembagian sebesar 80:20. Semua proses ini diperlukan dalam rangka membuat model yang baik.

### Cleaning
Dikarenakan data tidak terdapat nilai null dan data duplikat tidak banyak maka tidak dilakukan cleansing. 

### Feature Engineering
Berdasarkan gambar 3b, terlihat bahwa terdapat korelasi tinggi antara kolom panjang dan lebar kelopak(petal). Oleh karena itu, dari _insight_ tersebut bisa dijadikan masukan untuk membuat sebuah fitur baru. Salah satunya adalah dengan menghitung ratio antara kolom panjang dan lebar kelopak(petal), yang harapannya dapat memberi _impact_ pada hasil evaluasi model.

### Feature Extraction

Diketahui bahwa pada gambar 3a, terdapat beberapa kolom yang memiliki distribusi yang tidak linear. Maka dari perlu sebuah teknik yang dapat memberi standar pada fitur yang memiliki distribusi yang tidak linear. Salah satu teknik yang dapat digunakan adalah _StandardScaler_ dimana teknik ini digunakan untuk menstandarkan fitur dengan menghilangkan rata-rata dan menskalakan ke varians unit. _StandardScaler_ memastikan bahwa setiap fitur memiliki mean 0 dan standar deviasi 1. Ini penting untuk algoritma machine learning yang sensitif terhadap skala data. Lalu manfaat lain adalah menghindari dominasi fitur. Fitur dengan skala yang lebih besar bisa mendominasi model dan menyebabkan model menjadi bias terhadap fitur tersebut. Dengan menstandarkan fitur, maka dapat dipastikan bahwa semua fitur berkontribusi secara setara dalam proses training model.
