## eye-attendion-detection-opencv
### Kamera Görüntülerinden Dikkat Tespiti
OpenCV ile görüntü işleme tekniği ve Python programlama dili kullanılarak kamera üzerinden kişi 
yada kişiler için dikkat tespiti yapıldı. 
Kameradan yüz ve gözlerin tespiti gerçekleştirildi, kişinin ekrana doğru bakıp bakmadığı tespit 
edilerek dikkatli yada dikkatsiz sınıflandırması yapıldı.
Kameradan yüzün ve gözlerin tespiti için OpenCV kütüphanesini aşağıda belirtilem modelleri 
kullanıldı.
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
eye = cv2.CascadeClassifier("haarcascade_eye.xml"); 
### Dataset: 
OpenCV kütüphanesi kullanılarak göz tespiti sonrasında kameradan dikkatli ve dikkatsiz 
örneklerinin herbiri için 100’er adet görüntü alındı. Alınan bu görüntüler dataset dosyasına, label 
verilerek kaydedildi. Dikkatli için 1 Label’i kullanıldı. Dikkatsiz örnekler için 2 Label’i kullanıldı.
### Training:
Dataset dosyasından alınan her bir görüntü ve Label Id’leri için eğitme işlemi yapıldı. Eğitme 
işlemi sonrasında oluşturulan model, trainer dosyasının içine trainer.yml olarak kaydedildi.
### Recognition: 
Kamera üzerinden yüzlerin ve her bir yüzün gözleri için yine openCV ile tanıma işlemi 
yapıldı. Tanınan gözler ile model kullanılarak tahmin(predict) işlemi yapıldı. Yapılan tahmin ile 
gözlerin ait olduğu sınıf , yüzün çerçevesi üzerine yazıldı.
### Proje çıktı örnekleri aşağıdaki gibidir.
![alt text](https://github.com/htcoztrk/eye-attendion-detection-opencv/blob/master/output_example.PNG "Logo Title Text 1")

