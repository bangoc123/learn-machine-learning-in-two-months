### 4. Bài toán hồi quy (Regression)
----

   - **a. Mô hình Hồi quy tuyến tính (Linear Regression Model)**
        - [Bài toán đơn giản](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/models/linear-regression/README.MD/#1-b%C3%A0i-to%C3%A1n-%C4%91%C6%A1n-gi%E1%BA%A3n)
        
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

            ![](https://lh3.googleusercontent.com/oobRXwN--zTmadkkQyxRGc-q4D-4L0yqXuuODnztnfm7eiXaKD0Tu91Xawe93Z4UZOa99aklH1N3gJx0TnFGnex0VWrpwaDKd0iI8kPhHECJqK-pzccxZq_uWd9JX1W1S8FDjx-s9MK9CFHx-NWBZhx2AkNCz8hcjirbUu8Y1ucju4uHhpbsOdswW6E8iQLKa3P0jkQcBGrg5yvrC2AAm7tTpT82qaRx-ZX4cGS9Mg3fUwDBnXCADwiNg4pm3CbMAYxDRCES6UiO2ZSerzuSriUMO92ShE1T0TlWTDEYIcFLsyLB6Q7kgK0JtEQpGGauko_624BHZkoa28C7EnB9nTG8v9PcN0Y9RWBxsmvcBN2-lOkh2v7WIuEilp2uhvMvsZHF0yO3qUWQtvFrIoiR8MnlYLUaORpkxtOC3Adv4kFDdIR-yqAiHGoFMG_LoUSa_YQlisbX2xOym_FODieBiMMmfuTtMCeEBXq73-Nfi3skuRDnHYzATRnyOQv3aam0Bo-D8ufVOTrm64Rmw59EmZ9DMvnL3VWUOX1VHqRHDNZzc5aX8fKhXeOznYe4CW_P2mqyuRZUoL9F7SdHnWkc0KKHUNZwOHIAQGl3NQJkZq-c9YRV5gOfOesV5SW5Dj2LynHEd4ysqBP8CFd4lQsiut-ZqL2nnDM72sdF78GHrEEU-XpU4RKu5QnN=w516-h324-no)
            
            Bài toán của chúng ta là phải đi tìm hàm **F** thể hiện được mối quan hệ giữa **Y** và **X** có dạng **Y= aX + b**. Chú ý **X** và **Y** đều là ma trận/vector. 
            
            ![](https://lh3.googleusercontent.com/-hgYhecmCuMDgkkr79qCq-D8QQWcK7Vgv5gL1eLAGdqtXAg_w-RIniTHgxfzb4i7qUhMJjdZdkkt6xVfLl6oQkR2dD_QUuDY0iz-LPZiebyXl_4TZ1pQ63fHsWgImjEIiAwKeR9R8_h63sUO1ky2OmaHJX7Ba8qdWF9R-WqPjoz4WCd2DFsb_EE3dxQl9fCzgSt4-u6cJudvAbu5dFzor77hgKyAiHWQyT6ynXLTG5yTZjstdxhDHDHE-tq6Ru7zR_ToEZHH1PzgomaPybfZ7yZjaXUX36pzxtbjh7X_RuLAlnDPxA3ZQZb1G9BDpBZrXDAKjPk9HbHt0zxgPXza3EcTTJNnk3h7DtRWP3vgREmIJJp83P7QLlKDaYOdbs3QO77aFRTtHPUShVJx9t-nuYPW2vpzNJRnOnioLWr1CFzc6KdStczbAWuY8iKRXZ7dN6_fvzdczqe4WdMMAtQyKPsT8oKHNnVnQwl0qDN00UPCdCpdoxj75vQXOylw77Aerh9q8l35sELIzK9mDFh4GbTjYunX5Fj37GBAMst8udjPncxUcQ8qhAvg8U9kZGcp-OGiNROySfXtF4eVMHZ4YFfvpyeSq5PksFyEkN97K62PYyZu4BcGrXZGqzGg9HHC4uWkqWPoyW8pzMw5b-D5Y81az68Az8NANFTY3h65crI-1FWBH162JOHw=w537-h336-no)

        - Định nghĩa
            
            Trong thống kê, hồi quy tuyến tính là một phương pháp dùng để mô hình hóa mối quan hệ giữa một đại lượng vô hướng với một hoặc nhiều biến độc lập.
               
            Các dạng của mô hình hồi quy tuyến tính:
            
            <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;h_\theta&space;(x)&space;=&space;\theta_1x_1&space;&plus;&space;\theta_2x_2&space;&plus;&space;\theta_3&space;=&space;\begin{bmatrix}\theta_1&space;\&space;\theta_2&space;\&space;\theta_3&space;\end{bmatrix}&space;\begin{bmatrix}&space;x_1\\&space;x_2\\&space;1&space;\end{bmatrix}&space;=&space;\theta^TX" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;h_\theta&space;(x)&space;=&space;\theta_1x_1&space;&plus;&space;\theta_2x_2&space;&plus;&space;\theta_3&space;=&space;\begin{bmatrix}\theta_1&space;\&space;\theta_2&space;\&space;\theta_3&space;\end{bmatrix}&space;\begin{bmatrix}&space;x_1\\&space;x_2\\&space;1&space;\end{bmatrix}&space;=&space;\theta^TX" title="h_\theta (x) = \theta_1x_1 + \theta_2x_2 + \theta_3 = \begin{bmatrix}\theta_1 \ \theta_2 \ \theta_3 \end{bmatrix} \begin{bmatrix} x_1\\ x_2\\ 1 \end{bmatrix} = \theta^TX" /></a>
            
            <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;h_\theta&space;(x)&space;=&space;\theta_1x_1&space;&plus;&space;\theta_2x_2&space;&plus;&space;\theta_3x_1x_2&space;&plus;&space;\theta_4x_1^2&space;=&space;\begin{bmatrix}\theta_1&space;\&space;\theta_2&space;\&space;\theta_3&space;\&space;\theta_4&space;\end{bmatrix}&space;\begin{bmatrix}&space;x_1\\&space;x_2&space;\\x_1x_2&space;\\x_1^2&space;\end{bmatrix}&space;=&space;\theta^TX" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;h_\theta&space;(x)&space;=&space;\theta_1x_1&space;&plus;&space;\theta_2x_2&space;&plus;&space;\theta_3x_1x_2&space;&plus;&space;\theta_4x_1^2&space;=&space;\begin{bmatrix}\theta_1&space;\&space;\theta_2&space;\&space;\theta_3&space;\&space;\theta_4&space;\end{bmatrix}&space;\begin{bmatrix}&space;x_1\\&space;x_2&space;\\x_1x_2&space;\\x_1^2&space;\end{bmatrix}&space;=&space;\theta^TX" title="h_\theta (x) = \theta_1x_1 + \theta_2x_2 + \theta_3x_1x_2 + \theta_4x_1^2 = \begin{bmatrix}\theta_1 \ \theta_2 \ \theta_3 \ \theta_4 \end{bmatrix} \begin{bmatrix} x_1\\ x_2 \\x_1x_2 \\x_1^2 \end{bmatrix} = \theta^TX" /></a>
            
            Công thức tổng quát:
            
           <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;h_\theta(x)&space;=&space;\sum\limits_{i=0}^n&space;\theta_ix_i&space;=&space;\theta^Tx" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;h_\theta(x)&space;=&space;\sum\limits_{i=0}^n&space;\theta_ix_i&space;=&space;\theta^Tx" title="h_\theta(x) = \sum\limits_{i=0}^n \theta_ix_i = \theta^Tx" /></a>
        
        
   - **b. Hàm mất mát (Loss Function)**
   
        - Định nghĩa
        
        Hàm mất mát trả về một số không âm thể hiện **mức độ chênh lệch** giữa giá trị mà model của chúng ta **dự đoán** và giá trị **thực tế**.
        
        - Ordinary Least Squares
        
        **Mức độ chênh lệch** như chúng ta nhắc bên trên chính là tất cả các đường màu xanh được biểu diễn dưới đây. Ta sẽ có một hàm <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;J(\theta)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;J(\theta)" title="\large J(\theta)" /></a> thể hiện **trung bình tổng các đại lượng sai lệch**.
        
        ![](https://lh3.googleusercontent.com/p8NiIQCAE_2gPu5zq-KS5NCDW6zDYaUV8a6kwsrEi32fh3cvKK5ED1DNguQddldnXtBQ7j8z9Ff-C91zTa6y8eSMheFhudoeysOFlGh8hsCaZjLZQsSSHcLPCawjSbVxcNQMZgTvSLve3qX0hAerpp33nX7udhXyxLWFf39sV6bbdgHc17Y6ii4hr9khLRq6y8G_CznbXv1SPSYAiTzq-fu8lSnkrOQIaQcf7Adx5bJSYUQsOflZG7UQvdGbo7PuWkveKty0V8d9_aAuMhm49HvOw5aDYC2Ev5UHe6sC_Y1JGFqe2kQB_nPDLWWURcPh3asEMnuJxyXomkNFgotZC2DIM7HR1ut_357rZmU29p7qIpYn4UckJ4OoGu8DDzuIId896d9sAjPd7_uuB8ChFR6Ke0TgrpG3DsyHCE4EdDtfmxeY3o5t-G9ilDXZ8r2KgqqWyxdS8LdtkNHGXk7xiMLr9YZSOP6QnHv4gSAJVHSHbu_2RYXhvTm9GvoaYs05lltFQ3wYf46ZQQ7FDDP8Fj1abHx-MdxsSz9_6yU6Hgy9xHDRJdPg8AP1pm51UcVJdGFS3dOku8hz7JbZf14QK7xcd8yehSWK5OMZVcc78loUy6LyS4iofJPv01geJ8VgdhnYAlvmdvoxgVrpZFFPIRtWzA8X8SqdQsnsyFOb4Jx-TfISdy5oIk7ESA=w551-h335-no)
        
        Công thức:
        
       <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;J(\theta)&space;=&space;\frac{1}{2m}&space;\left&space;\|&space;X\theta&space;-&space;y&space;\right&space;\|^2&space;=&space;\frac{1}{2m}&space;\sum\limits_{i=1}^m(h_{\theta}(x^{(i)})&space;-&space;y^{(i)})^2&space;=&space;\frac{1}{2m}&space;\sum\limits_{i=1}^m(\theta^Tx^{(i)}&space;-&space;y^{(i)})^2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;J(\theta)&space;=&space;\frac{1}{2m}&space;\left&space;\|&space;X\theta&space;-&space;y&space;\right&space;\|^2&space;=&space;\frac{1}{2m}&space;\sum\limits_{i=1}^m(h_{\theta}(x^{(i)})&space;-&space;y^{(i)})^2&space;=&space;\frac{1}{2m}&space;\sum\limits_{i=1}^m(\theta^Tx^{(i)}&space;-&space;y^{(i)})^2" title="J(\theta) = \frac{1}{2m} \left \| X\theta - y \right \|^2 = \frac{1}{2m} \sum\limits_{i=1}^m(h_{\theta}(x^{(i)}) - y^{(i)})^2 = \frac{1}{2m} \sum\limits_{i=1}^m(\theta^Tx^{(i)} - y^{(i)})^2" /></a>
        
        Vậy để hàm **F** càng xấp xỉ data của chúng ta thì hàm <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;J(\theta)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;J(\theta)" title="\large J(\theta)" /></a> chúng ta phải đạt giá trị nhỏ nhất theo **<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a>**. 
        
   - **c. Thuật toán tối ưu Loss Function (Optimization Algorithms)**
   
        - Normal Equation
        
        Một liên tưởng đơn giản, như chúng ta học từ cấp 3, cách đơn giản nhất để tìm giá trị nhỏ nhất của một hàm số (cực tiểu), ta sẽ cho đạo hàm bằng 0 và tìm **<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a>**. 
        
        
        - Công thức toán học
        
            Tìm đạo hàm của <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;J(\theta)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;J(\theta)" title="\large J(\theta)" /></a>    
            
            Một chút biến đổi:
            
            <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;J(\theta)=\frac{1}{2m}\left&space;\|&space;X\theta-y&space;\right&space;\|^2=\frac{1}{2m}(X\theta-y)^T(X\theta-y)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;J(\theta)=\frac{1}{2m}\left&space;\|&space;X\theta-y&space;\right&space;\|^2=\frac{1}{2m}(X\theta-y)^T(X\theta-y)" title="\large J(\theta)=\frac{1}{2m}\left \| X\theta-y \right \|^2=\frac{1}{2m}(X\theta-y)^T(X\theta-y)" /></a>
            
            Chúng ta đơn giản hàm bằng cách bỏ đi <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\frac{1}{2m}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\frac{1}{2m}" title="\large \frac{1}{2m}" /></a> vì cuối cùng chúng ta cũng cho đạo hàm bằng 0.
            
            <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;J(\theta)=((X\theta)^T-y^T)(X\theta-y)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;J(\theta)=((X\theta)^T-y^T)(X\theta-y)" title="\large J(\theta)=((X\theta)^T-y^T)(X\theta-y)" /></a>
            
            <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;J(\theta)=(X\theta)^TX\theta-(X\theta)^Ty-y^T(X\theta)&plus;y^Ty" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;J(\theta)=(X\theta)^TX\theta-(X\theta)^Ty-y^T(X\theta)&plus;y^Ty" title="\large J(\theta)=(X\theta)^TX\theta-(X\theta)^Ty-y^T(X\theta)+y^Ty" /></a>
            
            Chú ý vì <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;X\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;X\theta" title="\large X\theta" /></a> và <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;y" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;y" title="\large y" /></a> đều là vector nên khi chúng ta nhân chúng lại với nhau, vị trí của chúng trong tích không quan trọng nữa. Cho nên:
            
            <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(X\theta)^Ty&space;=&space;y^T(X\theta)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;(X\theta)^Ty&space;=&space;y^T(X\theta)" title="(X\theta)^Ty = y^T(X\theta)" /></a>
            
            Cho nên:
            
            <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;J(\theta)=\theta^TX^TX\theta-2(X\theta)^Ty&plus;y^Ty" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;J(\theta)=\theta^TX^TX\theta-2(X\theta)^Ty&plus;y^Ty" title="\large J(\theta)=\theta^TX^TX\theta-2(X\theta)^Ty+y^Ty" /></a>
            
            Tiến hành đạo hàm:
            
            <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\frac{\partial&space;J}{\partial&space;\theta}=2X^TX\theta-2X^{T}y=0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\frac{\partial&space;J}{\partial&space;\theta}=2X^TX\theta-2X^{T}y=0" title="\large \frac{\partial J}{\partial \theta}=2X^TX\theta-2X^{T}y=0" /></a>
            
            Tương đương:
            
            <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;X^TX\theta=X^{T}y" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;X^TX\theta=X^{T}y" title="\large X^TX\theta=X^{T}y" /></a>
            
            Giả sử <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;X^TX" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;X^TX" title="\large X^TX" /></a> có thể nghịch đảo, chuyển vế ta có:
            
            <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta=(X^TX)^{-1}X^Ty" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta=(X^TX)^{-1}X^Ty" title="\large \theta=(X^TX)^{-1}X^Ty" /></a>
            
            
            
            - Điểm mạnh: Công thức đơn giản và chỉ cần đổ data vào tính toán
            - Điểm yếu: Tài nguyên để tính toán ma trận nghịch đảo với lượng data lớn là rất tốn kém nên trong thực tế các bài toán hồi quy tuyến tính rất ít khi sử dụng phương pháp này
        - Gradient Descent
        
            <img src="https://lh3.googleusercontent.com/kSRGUYFjiH6o14T8Vh0OjWtlMhpuvjcEHmaJUZ3XZKRIaZWVfjJDkFxHaJYPFkc1ih1ti7mX7V5EU70w46qzQzxTz6P15bCotjVCn2RfHFPFtyTyFbZwRGFeLvhiGR41AIrpZSG-ZjEdxSn1jDQ8xa7vkKJXzOgdrQVVNT5x8kUM8fWh9X0aFpNHeHJJbIFbf3sBLDZKejzqI8s0fTR3CLZCcJCayzGu81IXS59BhzPD6lFl0hXjo6yItd3Ff8zMgipm3_J7pn06J5xxkb_MEzwm3lsynzQwv1sbHx-aRZK2THa7n53vi1btjfr7PuTbt4qiviqaOFvE3p1takek_xaq_FHQzwtTwHJnvsyUuc-fPP3sNxf5cKSydapub-DFWmuAvqaSVq8MmDiqMzznlkynODcYK2fQWgBuYREl4BKoe-vUU7E-4_Plet0lA8ji1lohWMQX9EtYn0qYJlJXXtkT6hoL27eCbq7BfzFKJrPn3ROufbjbq3koCjT_BXLmSfQga6Gue4h8CbSCUk7Pq1WleAF1T9J25hUDT-0ToDMvcKjyY3bwOh4h8RcL_hgGRLmZ1ydFJUsVXM4GM0hI7gWmGEfBN2QWtXdSX7I8s8Y_RLDBsjOOPwGrWKX4MtoPwSlE_Ea_bMJCQEgvIhtV8HEt7mtErdnVBy8yLF1w4huwdZkd6RhiGpI8Z0Y2q6J3UCFegyqVYJHHSEquCpg=w729-h284-no" >
            
             Thay vì sử dụng công thức ăn liền Normal Equation, thì trong thực tế chúng ta sẽ sử dụng thuật toán Gradient Descent. Giải thích một cách đơn giản, chúng ta sẽ cho <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a> tăng và giảm một khoảng nhất định, sao cho giá trị của hàm Loss function <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;J(\theta)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;J(\theta)" title="\large J(\theta)" /></a> giảm dần đến giá trị cực tiểu. 
             
             Như bạn có thể thấy trên hình: Lúc ban đầu hàm <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;J(\theta)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;J(\theta)" title="\large J(\theta)" /></a> có giá trị lớn nên hàm ban đầu cần tìm dự đoán chưa chính xác các dữ liệu, nhưng khi hàm <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;J(\theta)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;J(\theta)" title="\large J(\theta)" /></a> giảm dần, hàm ban đầu dự đoán chính xác hơn rất nhiều.
             
            - Công thức toán học
               
                
                <img src="https://lh3.googleusercontent.com/7QSSTJjZQmKBF0JvW5uuRji4sFgnP3IsINkFFwIjVN5ed6WpkjrUjD9Y-r8neeM_7B5XpdyGx1-Z3gTwfocaSniYvaXZ2OJCfk3jckY4bEf1Bhr34SZw63dxhgp_29VWJLJiO12mx8m34Qz_05pAhkYXCcAJMnfo7p-2SIEtNXPoxlCF9xCY9IQ7yZZ8CSDOIKawxrgn6CkjeLAApbLzzGRIVuduvfUo2oa_guYn9gsB9rAtz2WkASHRrdYMa3DLlFU8womgizVku6YkNIT8sf_fTKGhti5bi_zfUVL0FfYkCE6obLKFeRkhZjE3yfRfp7ZcyjJlzDAGcEoLiMIieBtqGcsJ4lbT2s5hstfPVKkqpx9Zd0YdD1lrZw8_6vhvWvKJiXhhcr3XbYfvhR95o5_cCzKdTOVQllw-lqJjVWRJfThg7PUpy8VTQtrAICVwDjgPSsaTQvookIClY35LLhy2J2xdXKw6hwqMc3050jIwrvHZf6oBVVFiPT7isvaBR5yXr80wQ-8QiUzNHaqa5BGTdZ6h2zQi1yWqoFRHltpHvXs3tN0leVj6pXw7uQLgNoadzXdRqbYQZNzjX7NUIgl1YPrV7WRQezPZEZaklKY5nEwuKdsDH9xbMjrb-DLQ9iwkCsXP4ZTV2KL2Lg9QVE3KBGOLdV7B3VPqeBYkG0vFdmogVPHw1FL78QNekX0sKE1O79P77mfVyTpP-nI=s708-no" >
                
               - Diễn giải công thức toán:
               
              <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta_{new}&space;:=&space;\theta_{old}&space;-&space;\alpha\triangledown&space;J(\theta)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta_{new}&space;:=&space;\theta_{old}&space;-&space;\alpha\triangledown&space;J(\theta)" title="\large \theta_{new} := \theta_{old} - \alpha\triangledown J(\theta)" /></a>
               
               Tương đương với:
               
               <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta_{new}&space;:=&space;\theta_{old}&space;-&space;\alpha&space;\frac{1}{m}X^T(X\theta&space;-&space;y)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta_{new}&space;:=&space;\theta_{old}&space;-&space;\alpha&space;\frac{1}{m}X^T(X\theta&space;-&space;y)" title="\large \theta_{new} := \theta_{old} - \alpha \frac{1}{m}X^T(X\theta - y)" /></a>
               
                Các bạn nên nhớ là <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a> là một vector. Việc cập nhật <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a> tương ứng với việc cập nhật từng phần tử của <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a>. 
                
                <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta_0&space;:=&space;\theta_0&space;-&space;\alpha&space;\frac{1}{m}\sum\limits_{i=1}^m(h_\theta(x^{(i)})&space;-y^{(i)})x^{(i)}_0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta_0&space;:=&space;\theta_0&space;-&space;\alpha&space;\frac{1}{m}\sum\limits_{i=1}^m(h_\theta(x^{(i)})&space;-y^{(i)})x^{(i)}_0" title="\large \theta_0 := \theta_0 - \alpha \frac{1}{m}\sum\limits_{i=1}^m(h_\theta(x^{(i)}) -y^{(i)})x^{(i)}_0" /></a>
                
                <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta_1&space;:=&space;\theta_1&space;-&space;\alpha&space;\frac{1}{m}\sum\limits_{i=1}^m(h_\theta(x^{(i)})&space;-y^{(i)})x^{(i)}_1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta_1&space;:=&space;\theta_1&space;-&space;\alpha&space;\frac{1}{m}\sum\limits_{i=1}^m(h_\theta(x^{(i)})&space;-y^{(i)})x^{(i)}_1" title="\large \theta_1 := \theta_1 - \alpha \frac{1}{m}\sum\limits_{i=1}^m(h_\theta(x^{(i)}) -y^{(i)})x^{(i)}_1" /></a>
                
                <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;..." target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;..." title="..." /></a>
                
                <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta_j&space;:=&space;\theta_j&space;-&space;\alpha&space;\frac{1}{m}\sum\limits_{i=1}^m(h_\theta(x^{(i)})&space;-y^{(i)})x^{(i)}_j" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta_j&space;:=&space;\theta_j&space;-&space;\alpha&space;\frac{1}{m}\sum\limits_{i=1}^m(h_\theta(x^{(i)})&space;-y^{(i)})x^{(i)}_j" title="\large \theta_j := \theta_j - \alpha \frac{1}{m}\sum\limits_{i=1}^m(h_\theta(x^{(i)}) -y^{(i)})x^{(i)}_j" /></a>
                
                
                Nếu bạn đã từng lập trình thì có thể tinh tế nhận ra cách tính toán này. Chúng ta sẽ tạo ra một vòng lặp và qua mỗi vòng lặp đó ta sẽ gán giá trị mới cho <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a>.
                
                
               
               - Chú ý: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\alpha" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\alpha" title="\large \alpha" /></a> được gọi là một hyper hyper parameters. Nó sẽ được đặt bởi một giá trị cho trước. Ví dụ mình hay đặt sử dụng 0.01 hoặc 0.0001. Ý nghĩa của nó là một tham số quyết định tốc độ học nhanh hay chậm của mô hình.
               
               
                    - Nếu <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\alpha" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\alpha" title="\large \alpha" /></a> quá lớn: Hàm không thể hội tụ và giá trị của nó sẽ nhảy đến vô cùng. Khi lập trình bạn có thể sẽ hay gặp (nan) trong Python.
                    
                    - Nếu <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\alpha" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\alpha" title="\large \alpha" /></a> quá nhỏ: Mô hình hội tụ quá lâu, tốn kém thời gian.
               
                    <img src="https://lh3.googleusercontent.com/5VKCn6gZi7lnlao9_2s8qLMckEs6_fffkqgKOPvT9fWolWyj9AD5CiMrp2H9Vs5UxAVB2QHM-Gr433jm1dQI97lNKmbq9a7_UlZsM0CoRTbwZDfWWjYYjN0nOZVz7UfF_Uuqnu6CMIJcBKIHKYSZiGFeSmDZdMXD30StDz6sLHNmJxdSFWanjXGlb4i8dCwYlxkjp547VPKOUPiEDjPbnltCAfeKu5KLJJWkjmE8RQ3BCg0q_Lr5Sqairdp7gCyfUNUx15UdTEAfqzY1BRdyYSmpDIm419KB5sZrAwUt4uOeSPZyy4I_R086nLFXkfNBMsBDI0rQQLRcJlaop7xuqzO3TMGCveWZVspNm3Cuy7QN_s60PqOCU_3T1Ao5O3RG4zmmqA7-WublIRsEE3dauRXlIxxq5WW2-Unpt1Z1JyihAd-XfHeU_plxaazMNC7NNqE2Cxe3An89gXqEYHxAcWDKhjuRHovL44QkXQCeS9gWZpTi51OGkgN1IKd1UguUohys4RpHy-QxtLiEL66sWBOq0mQqx6drNHIhiwENXsU_mRe8mL6gYmHb2FiTVXFRt4GTGLa2zKZM58dMgp59W_TfiXiSZvt_1nkGxQyDB1li8VoHyYNo1iKh6idoP5t_1SirOlkzsWdtND2KSUL5rweJ_JOP_DlTedVUR6iph-f0pJ3cPvQAcRS6KWCxsl-17dywHD4epVVf6Si2IEY=w950-h529-no" >
               
                
            - Điểm mạnh
            
                Tính toán nhẹ nhàng hơn rất nhiều so với phương pháp ban đầu. Về sau chúng ta sẽ tìm hiểu các phương thức khác kết hợp với Gradient Descent để giảm nhẹ khối lượng tính toán.
            
            - Điểm yếu
                
                Kết quả thường không chính xác 100%, nhiều vấn đề liên quan xảy ra ví dụ như giá trị của hàm Loss không thể giảm thêm mà bị mắc kẹt tại một điểm local nào đó. (Như hình mình họa bên trên)
                
                Có rất nhiều phương pháp để giải quyết vấn đề này, tôi sẽ trình bày ở các bài tiếp theo.
            
   - **d. Bài toán thực tế**
        
        - Bài Toán Grab
        
            Toàn bộ Solution nằm [tại đây](https://github.com/bangoc123/learn-machine-learning-in-two-months/blob/master/models/linear-regression/GrabProblem/Grab-Solution.ipynb)
        
            - 1. Mô phỏng data
            
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

                Ta sẽ phải mô phỏng data này bằng code.

                Gọi Số KM là <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;X" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;X" title="\large X" /></a> và số tiền cần phải trả là <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;Y" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;Y" title="\large Y" /></a>. Gọi <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\hat{Y}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\hat{Y}" title="\large \hat{Y}" /></a> là hàm chúng ta sẽ dự đoán. 
                
                <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\hat{Y}&space;=&space;X_0\theta_0&space;&plus;&space;\theta_1&space;=&space;\begin{bmatrix}&space;X_0\&space;1&space;\end{bmatrix}&space;\begin{bmatrix}&space;\theta_0&space;\\\theta_1&space;\end{bmatrix}&space;=&space;X\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\hat{Y}&space;=&space;X_0\theta_0&space;&plus;&space;\theta_1&space;=&space;\begin{bmatrix}&space;X_0\&space;1&space;\end{bmatrix}&space;\begin{bmatrix}&space;\theta_0&space;\\\theta_1&space;\end{bmatrix}&space;=&space;X\theta" title="\large \hat{Y} = X_0\theta_0 + \theta_1 = \begin{bmatrix} X_0\ 1 \end{bmatrix} \begin{bmatrix} \theta_0 \\\theta_1 \end{bmatrix} = X\theta" /></a>
                
                **Việc chúng ta cần phải làm là tìm** <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a>
                
                Mô phỏng X0: 
                
                ```
                X0 = np.array([[2], [7], [9], [3], [10], [6], [1], [8]])
                
                ```
                
                Tạo X: X là một ma trận kích thước (8, 2) được ghép bởi X0 kích thước (8, 1) và một vector **ones** kích thước (8, 1) tất cả các giá trị là 1.
                
                ```
                ones = np.ones_like(X0)
                
                # Nối 2 vector/ma trận theo chiều dọc
                
                X = np.concatenate((X0, ones), axis=1)
                ```
                
                Giá trị của X sẽ là:
                
                ```
                array([[ 2,  1],
                   [ 7,  1],
                   [ 9,  1],
                   [ 3,  1],
                   [10,  1],
                   [ 6,  1],
                   [ 1,  1],
                   [ 8,  1]])
                ```
                
                
                Mô phỏng Y:
                
                ```
                Y = np.array([[13], [35], [41], [19], [45], [28], [10], [55]])
                
                ```
                
                Giá trị của Y sẽ là:
                
                ```
                array([[13],
                   [35],
                   [41],
                   [19],
                   [45],
                   [28],
                   [10],
                   [55]])
                ```
                
                Sử dụng **matplotlib** mô phỏng các dữ liệu với hàm ``` plt.scatter(X_begin,Y); plt.show()```
                
                <img src="https://lh3.googleusercontent.com/tk2v2HtHUA6HSLMlMpBD7JR-558fqnn7y9RzMv3Da6xeVAnwNeQdRmMkvD2DZGsIVdef30yTU11Nh1solD4aWPH8e1O5L8BreENmjdiL9J_y5sGoPfHMZ_V_SfBwzcX-SxembeX2N50725zAo14ofy3wmKna8amM2d7q4RtHrnNUG9ra22DfqgXcFLWsajLrRFAKQ9iC9TNv4-U-BniiqXfyLCAj2qPhGOtvXPUDfGhuyr7DPxtSOUzhmziqSA3BF3r3DG6BcgXvFUN-NkZ_uzlK2khXKZERzprH1IA2E0T6HxptZBURkb1qrSaGVsUc72t_74FUNxHXlhJa342wVcyd-TrQEWyX83FQVY-9m4SAk3TcUmF53GZok-GemeWuOaJ26hoCfEvn3AOzY1mqieQeGUrZeItskLXpOkGAcDPjyaR7vGTAtDZZPNs9b4DFVcj1pjH3jwh9WIFLHThCtXXZtvt36bCwqmU1hm_Ot9GPzHiH0m6NYO2bWhyx7vifl7xtzVADpR5ijWq8yTEDmHhOwiLN9HI32DKnSX92uQEJGV_N8lnYHs-iy_4OFjZlf0kkrYq8FuOQpbQg9QFJ4vZOyAotn__f48iQO0mDk3Wk9QozG3dn5MQ1Z0net5mZxnww_VjpxwHI9Ntkz1RgxESrWTwXBYmopZjiXW04dSFzPBZaFEWIjx3nZ0MXrWXCBMb1uxgvP6Qvp4szpEc=w1218-h686-no" width=300>
                
            - 2. Sử dụng phương pháp Normal Equation.
                
                Tính trực tiếp <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a> bằng công thức: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta=(X^TX)^{-1}X^Ty" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta=(X^TX)^{-1}X^Ty" title="\large \theta=(X^TX)^{-1}X^Ty" /></a>
                
                ```
                
                # API np.linalg.inv() dùng để tính ma trận nghịch đảo
                
                theta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
                
                ``` 
                
                Chúng ta sẽ tính ra được kết quả của <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a>:
                
                ```
                array([[4.40880503],
                    [5.39937107]])
                
                ```
                
                Vậy <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\hat{Y}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\hat{Y}" title="\large \hat{Y}" /></a> có dạng:
                
                <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\hat{Y}&space;=&space;X\theta&space;=&space;\begin{bmatrix}&space;X_0\&space;1&space;\end{bmatrix}&space;\begin{bmatrix}&space;\4.40880503&space;\\\5.39937107&space;\end{bmatrix}&space;=&space;4.40880503X_0&space;&plus;&space;5.39937107" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\hat{Y}&space;=&space;X\theta&space;=&space;\begin{bmatrix}&space;X_0\&space;1&space;\end{bmatrix}&space;\begin{bmatrix}&space;\4.40880503&space;\\\5.39937107&space;\end{bmatrix}&space;=&space;4.40880503X_0&space;&plus;&space;5.39937107" title="\large \hat{Y} = X\theta = \begin{bmatrix} X_0\ 1 \end{bmatrix} \begin{bmatrix} \4.40880503 \\\5.39937107 \end{bmatrix} = 4.40880503X_0 + 5.39937107" /></a>
                
                Miêu tả dạng này:
                
                ```
                # See how it fits:
                plt.scatter(X_begin, Y)
                plot_polynomial(0, 10, theta)
                plt.show()
                ```
                
                <img src="https://lh3.googleusercontent.com/wuaN0MZAz7T_IRxO5EFl1C8LTfDAncNCCktbZZJHTzuc6rYNoZEQHFMOQZHVUES1XjDSHDrMDKxZut_qcCbBD9hQK42yfdBegZbOE0OwdYwUvAR_GAW_QeKljv84OBNRH3yj0A9GPww7kZ-n-50UacQA9Hx5ur_0M-5XEgScX8o7fDBi-jbkbk0BKAuOoM7r5oI-78g15b33Qer9eTAqXjcObqziwq_j_JhsLWOxgeMazTeCkMyTg3E_Zx09EYBT1XIs0cxgbmuTmmZ6gAc6YCWsfapWmcd-NBfdoz90JrvbgDQTaIJ5i2LIkgrMpT6kiuYt0g-RfcY9UDGnnwIXO9wy02sTFxg2dBjsuIc0KOVw1rKYIcBLwEh5ZbekdljAn4pMh6TWuFMXulvnenU9ng4vm3SALhr8FpZram65tjHLy1W4FuKS7rPGdCfGg1MwR9rEo3LvmkKKsdoUlSuZjyNpKRkixu-JTFHS1yc9oib4Iz7J5sLaT9hiaPB5u2xmguORP6YxRjc9wWh4yC5doEiiJSLrRg766f4QqgNEqxc2QXvyGBfpxt4gUwFN99fI1d2Ivg2YTBqp1ItYzA69Mr70f91RRA-janAxudZCrXmxrPHQOlIyMBQOAPk-e61dk4zXTgyXTbarg9fE7_kN75c-_i9h5jWfrsiLNS8JkTKNc3Aj-Kmip668OXqOSOuF-RZ1-KMY9WzjTN3Fklc=w884-h590-no" width=300>
                    
                 Chúng ta có thể thấy hàm <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\hat{Y}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\hat{Y}" title="\large \hat{Y}" /></a> (đường màu vàng) đã xấp xỉ chính xác nhất data.
                
            - 3. Sử dụng phương pháp Gradient Descent.
            
                Khởi tạo <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a> ngẫu như theo phân bố chuẩn.
            
                ```
                theta_gd = np.random.normal(size=2).reshape([2,1])
                
                ```
                
                Giá trị của <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a>:
                
                ```
                array([[-1.31237748],
                [ 0.22471167]])
                ```
                Đặt learning Rate <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\alpha" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\alpha" title="\large \alpha" /></a>:
                
                ```
                learning_rate = 0.02
                ```
                
                Tạo hàm tính gradient theo công thức <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\triangledown&space;J(\theta)&space;=&space;\frac{1}{m}X^T(X\theta&space;-&space;y)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\triangledown&space;J(\theta)&space;=&space;\frac{1}{m}X^T(X\theta&space;-&space;y)" title="\large \triangledown J(\theta) = \frac{1}{m}X^T(X\theta - y)" /></a>:
                
                ```
                def grad_cal(X, Y, theta_gd, m):
                    """
                    X: X's value
                    Y: Y's value
                    theta_gd: theta's value
                    m: number of samples
                    """    
                    g = 1/m * X.T.dot(X.dot(theta_gd) - Y)
                    return g.reshape(theta_gd.shape)
                
                ```
                
                Tạo hàm tính loss function theo công thức <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;J(\theta)&space;=&space;\frac{1}{2m}&space;\left&space;\|&space;X\theta&space;-&space;y&space;\right&space;\|^2&space;=&space;\frac{1}{2m}&space;\sum\limits_{i=1}^m(h_{\theta}(x^{(i)})&space;-&space;y^{(i)})^2&space;=&space;\frac{1}{2m}&space;\sum\limits_{i=1}^m(\theta^Tx^{(i)}&space;-&space;y^{(i)})^2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;J(\theta)&space;=&space;\frac{1}{2m}&space;\left&space;\|&space;X\theta&space;-&space;y&space;\right&space;\|^2&space;=&space;\frac{1}{2m}&space;\sum\limits_{i=1}^m(h_{\theta}(x^{(i)})&space;-&space;y^{(i)})^2&space;=&space;\frac{1}{2m}&space;\sum\limits_{i=1}^m(\theta^Tx^{(i)}&space;-&space;y^{(i)})^2" title="J(\theta) = \frac{1}{2m} \left \| X\theta - y \right \|^2 = \frac{1}{2m} \sum\limits_{i=1}^m(h_{\theta}(x^{(i)}) - y^{(i)})^2 = \frac{1}{2m} \sum\limits_{i=1}^m(\theta^Tx^{(i)} - y^{(i)})^2" /></a>
                
                ```
                def loss(X, Y, theta_gd, m):
                    """
                    X: X's value
                    Y: Y's value
                    theta_gd: theta's value
                    m: number of samples
                    """
                    return 1/(2*m) * np.sum((X.dot(theta_gd) - Y)**2)
                ```
                
                Tiến hành tạo vòng lặp, qua mỗi vòng lặp sẽ tính lại gradident đồng thời cập nhật lại <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a>
                
                ```
                for i in range(10000):
                    grad_value = grad_cal(X, Y, theta_gd, m)
                    theta_gd = theta_gd - learning_rate*grad_value
                    print(loss(X, Y, theta_gd, m))
                ```
                
                Giá trị của hàm Loss:
                
                ```
                890.0619039221115
                33.23288932117509
                19.912233421433992
                19.674950415271216
                19.640872759091934
                19.61022229452672
                19.579897901246316
                19.54984900715867
                19.520072377582952
                19.49056553425761
                ```
                
                Chúng ta có thể thấy lúc đầu khi <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a> có giá trị random, giá trị hàm Loss Function rất lớn. Sau đó khi <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a> được cập nhật thì giá trị hàm Loss Function giảm dần và hội tụ đến điểm cực trị.
                
                Ở vòng lặp cuối cùng 10000 giá trị <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a> thu được là: 
                
                ```
                [[4.40880503]
                [5.39937107]]
                ```
                Vậy hàm <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\hat{Y}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\hat{Y}" title="\large \hat{Y}" /></a> của chúng ta có dạng:
                
                <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\hat{Y}&space;=&space;X\theta&space;=&space;\begin{bmatrix}&space;X_0\&space;1&space;\end{bmatrix}&space;\begin{bmatrix}&space;\4.40880503&space;\\\5.39937107&space;\end{bmatrix}&space;=&space;4.40880503X_0&space;&plus;&space;5.39937107" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\hat{Y}&space;=&space;X\theta&space;=&space;\begin{bmatrix}&space;X_0\&space;1&space;\end{bmatrix}&space;\begin{bmatrix}&space;\4.40880503&space;\\\5.39937107&space;\end{bmatrix}&space;=&space;4.40880503X_0&space;&plus;&space;5.39937107" title="\large \hat{Y} = X\theta = \begin{bmatrix} X_0\ 1 \end{bmatrix} \begin{bmatrix} \4.40880503 \\\5.39937107 \end{bmatrix} = 4.40880503X_0 + 5.39937107" /></a>
                
                Biểu diễn hàm:
                
                 <img src="https://lh3.googleusercontent.com/Wh4KdEOh264CwdYMxVCkwxvjxeaIklKJgSGrPjmZ6A500RTMD4WcPmPTucGBwjP5h68wgkpZ5iF-gKOa-Cqsvp-JZbUkq3lxVPCyQ48OqSgmtn_St1nRX18ynpkgFpJSlL9qUPFcaaqXE7AAvFgI-JBMT2leTTeZ9dkHfnVk3HepkTxT15KUV0tFWeM7o5fBoe_JNtRNy3TAMsD79W_iiXE29TuaoCYySBpEjbH8Ibm-WCyJwVjmwPUrCAPG6JFlVNiTYlNjT6sCF50P9rUAvZ3G4e0FWukxfPfhfnOH4ZlG56dsABRtc7aX5quGAzKkFnSUqXmPVEHSudZhr05cYo3w_Esc4LtrOQDAOGXbusYxtIOqbe2TGyuU1dxRY1DyfxFog4pQ0TUF6BrImuCsiMTihPv3DiMzK1gKUgt08w-AdkHS3uzyZtPvUEM5Epk9IMC6m0dNlZzEluLKO64UdrZTNeTum_2kzFMaH5Za7x2-SP931xK2Yf7PENImpmcsWHASEid25bmU69cOgmnSih2NAF7JtqBJHlILcxcPtGtfi9wLrD-Qy6I4xlyaJ_oxLlGHwUkzlj3D5EWDNLNBTZUFJsAxxK6V-7ztkyPjBWKe4BJtOO7oNb981yAACMbFcozM18EcZBSwvOpicv-H8qyqICRvpMQyAiXLRu_H0a0ga_YpdiGLyoi8-sRhJBP1at8YGx_SqRWrYCVYczA=w844-h556-no" width=300>
                
                Chúng ta có thể thấy trong bài toán này, hai kết quả ra tương đồng nhau.
                
            - 4. Dự đoán
            
                Như 2 phương pháp trên chúng ta đã nắm được dạng hàm của <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\hat{Y}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\hat{Y}" title="\large \hat{Y}" /></a>. Để dự đoán vỡi những data mới. Chúng ta chỉ cần tạo hàm mô phỏng:
                
                ```
                def predict(X):
                    return X.dot(theta)
                ```
                
                a. Tôi muốn dự đoán số tiền phải trả khi tôi đi mất 20km
                
                ```
                # Chú ý: luôn phải nối 1 vào bên phải của vector đầu vào giống như ta đã training.
                X_new_1 = np.array([[20, 1]])
                predict(X_new_1)
                ```
                
                Ta thu được kết quả:
                
                ```
                [[93.5754717]]
                -> Số tiền cần phải trả chính là 93.5754717 (đơn vị 1000 đồng)
                ```
                
                b. Tôi muốn dự đoán số tiền phải trả khi tôi đi mất 40km
                
                ```
                X_new_2 = np.array([[40, 1]])
                predict(X_new_2)
                ```
                
                ```
                [[181.75157233]]
                -> Số tiền cần phải trả chính là 93.5754717 (đơn vị 1000 đồng)
                ```
            
            
        - Bài toán đo lường Hiệu quả sản xuất
            - Vấn đề
            - Giải quyết bằng mô hình Hồi quy tuyến tính
