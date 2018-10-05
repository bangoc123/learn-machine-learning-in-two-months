## Mô hình Hồi Quy tuyến tính

### 1. Bài toán đơn giản
----

Trước khi nhắc đến những công thức toán học dài dòng tôi muốn mô phỏng một bài toán đơn giản cho bạn dễ hình dung trước. 

Một hôm đẹp trời, tôi sử dụng ứng dụng Grab để đặt 1 chuyến đi từ nhà tôi đến Bờ Hồ Hoàn Kiếm dài **5km** nhưng xui thay ứng dụng này **mất khả năng tính toán thành tiền** và bác xe ôm đen đủi không thể biết được số tiền chính xác mà tôi cần phải trả.

May mắn thay, tôi vẫn còn truy cập được vào lịch sử các chuyến đi của mình như bảng bên dưới, liệu rằng dựa trên những thông tin này tôi có thể tính được số tiền mà tôi cần trả cho **5km** ngày hôm nay?

| Số Km (Km) | Số tiền cần trả (1000 VND) |
| ------ | ------ |
|    2   |   13   |
|    7   |   35   |
|    9   |   41   |
|    3   |   19   |
|    10  |   45   |
|    6   |   28   |
|    1   |   10   |
|    8   |   55   |


Giả sử **Số KM tôi đi** và **số tiền cần trả Grab** phụ thuộc tuyến tính vào nhau, tức là **số KM tăng** thì **số tiền cần trả Grab tăng** hoặc **số KM giảm** thì **số tiền cần trả Grab giảm**, tôi có thể tìm ra được 1 hàm biểu thị được mối quan hệ giữa 2 đại lượng này không? Câu trả lời là **Có**.

Tại sao lại cần tìm ra hàm đó? Vì nếu tìm ra được nó, việc tôi cần làm chỉ thay số KM tôi đi ngày hôm nay vào hàm và tìm ra được số tiền tôi cần trả.

Tôi đặt:
  - X là số Km tôi đi
  - Y là số tiền cần trả Grab
  
Hàm cần tìm sẽ có dạng **Y= aX + b**. Bài toán quy về: Với X và Y cho trước trên bảng trên, tìm 2 tham số **a** và **b**.

**Biểu thị data bằng đồ thị**

![](https://lh3.googleusercontent.com/hFRmPtUMXQCA-w2iS162eWwYWe_DGrY68HGN3VOeu0WpWegGycjVMsotWAv7hsIBOh0Uw0DOP0p9UF2CV754un3MlTA0EM3geMySyhTMHWP1JSoc6zekDOE-23QTiW4MADv8zDS_IsHEYPrQXcGuo1WsYKJGoqEj7i0iLTEFm_uFONyZNnZXr6hLu8zgcVgPMwEZABW6oSjW1lN6kH7P_4w46FdBenyZM6wyB1_ttrU0ZK1HJTkq8I0OUOw5Dmak7R_IDoNh_9S0xh35piemaAYAkz8xuqzRgX2QQ-K9dAxPyiueoIu1sIJ5VJHmtbVfJY6UkEfBhhAmnNNMVhfv-PtMNOmjCjk3EEDgIwv4AVGv3Xnm1LUQ_3XOcZ7ny20lccuJv2kRjtgVkPU2OxLcWBqbOzIcEKPX7R4OUs0OoK4ZXiDkScGQE89qn6OBRHCrFAnDq7eV9M8vig8Pq0J_3ednTwKvXdzEq29nbMwJFI64SdWLCov_vKfnoXmaI9dt0rtCCzK5nMzgXy38ujO6yWgGN8ojFPV85LFDJjJyW_S6sAcumNaKOY1YTyWlGatuY4s0o3LcZ24xS5-RHmNDvicNxYin8KJ2Rwqdxu95lOnQ0tTdTQXXQXBfFaGR_q9Ik6TBssL3f6Qj1LfxlbGb6VtkJ0Ola7stcGHJSl2OJgYgQrMieN2nyCkF=w510-h336-no)


