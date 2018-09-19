## Thị giác Máy tính (Computer Vision - CV)

### 1. Thị giác máy tính là gì?

<img src="https://lh3.googleusercontent.com/Ne6PiXk-ENbZtIRDlnBEewMKt359zBVKh76XfT02r3LyINhTYEk_gKiT8EI2iFl-7HI6zselTQqtbA6Zhq9NTIpafbJY76JQ3-CZGgLc0-4SUOClA6H-2fhyGA2OmpSSd1Fdlx9hv8t0yFgfCBu_5ZQi_UJ3VoO9N4w6Rc6aKZnbi80Drm0h4kP9jV3-QePnz-arEXPwPblGYVvHzdgv1wbbwEj1yDK12-6Bvmmd1fGTVGHq1QaTd0SMl729_dVC1LR-MK5fVHxjrlgXiTg983uvFIlR2RJGBpO6E3YIo3IcWeb0c-wi8ARxMge-4lEomJw3JRulNKZDsW0dVWivl59hvGZVaixotqH9Syh8EhClRTi57IBtm19CBJeWiSxyA-8YP9mBbfHKReRUUO4oObuMaJbgWdkp0cdY9Qa_coffvirbqx1X4DK01wlVvaJ3ey3xtmancNJ3Ib6p-p-oIH-A1afWkSys3XRJpuQ2wI7mIEJRx5TTjNVxWBX8bxpiw-1Xn1ySGZ23XnlixcVOtl8xZCEOuekqD6ygglGY1vlNrunY5CQWpfuwXiWrWiRKboExp49zXXqfXGfPZFXEyDXkiYANuoUlEhefZIxiPQOgHwRSis-H8ePl5tYLT3s2LVjXHEjxxIWvJ2hC0fNLEFFoOP-7t_WaVARCZnA6emyx_6jg3AFNdBcjQg=w263-h148-no" width="500">

Ảnh từ [CS231](http://vision.stanford.edu/teaching/cs131_fall1819/index.html)

Mục tiêu của ngành nghiên cứu này là viết ra những chương trình có thể hiểu được với hình ảnh.

Khởi đầu của môn học này là một project hè ở MIT.

![](https://lh3.googleusercontent.com/DYj-YAPnEdWNvGqj2sIXMnXccaTpoKnSc7nzpqy9DVWSXEJjVWbE0kCYHd8DeBs_7Y62uf7MTZIcYXFPvcFG8yqv4LCVr32K8XuXky3Cl31rWA3Qioc1LQUPZsVYe8GCPlaZgvoKC5r_yA_F0R2KGoz8ztymIOaZHtbTvp92rTjq28a9d9zcbzHllfE1BOiJUFKKfuIIBVntGlFyv2tPx-TXm1seV_fyRthbe0lLFlxpDGyW9gMH6Hqhmx23LM_zAdGILuyDNQTPt69PaEg59qNzNAMrN2CZb-axsia-kuxpuVoR3xJork0rmET5gRRjqNjyCJyEP09e4JQaoy2vd8b0_47IzI5LwstGtd9m-ZhIhczoNxPJAwkZlumi3ZshBhLhALk_HaUgzhWiQTQzMEDwhXwsoOGSQQ-RVEs7P-itUXgvYFPvi4UKEUE_ZT9s86D2McFZ-9xwc8MWkNTjpv9WS92jMs-dy5yLT7il5G8H1bjgZzfY9MaMprJbMdGMZFCtyWux3jYbZT864utZohco6PvHrUQ54qiFwrAtXO-1ntR54jV2AE1qAMdTrY_M5-oUjYtjojc_p6PRBO8peK9wIuBUMqcb1iYswB48UWoKfgiZJd0u4d2MRcTZEiNJapsMJBzcb0zgGBvY-VxxyouaPZ168BmPZQRPX73_s_2NKA_FYRZ0JubDMA=w813-h503-no)

Vậy những loại data nào có thể trích xuất từ một hình ảnh:
 
- Ma trận 3D
- Ngữ nghĩa của một hình ảnh

Ví dụ về Image Captioning

![](https://lh3.googleusercontent.com/NonYkmaZPybiw9SYRLHJIyqzmCEZPuE_aH3DtbvAXUdmEJUEs_us8WgQPDVn8AX8onliN0VEGUWEaTs3DWI9rThZ1TRL4V-UPSW2DPyL-oaypwnG_7TmyW30R0xJ645m4yPwPybnFFM4mOW3KuV6ApHxcjz9a_hRNl5JeFNXUagzVU2mXPNJaNXEOUtD4bHflfDwm0cQKwcGjXQqmKFhJtbQzH8OhNL4sDpB0mUb2zCiDwE9n2jIMKI97T4qu-yJb_SmQcyjPW7lQxjFsKXjycO1eboDIFWpI3DSDa9Z4gMurIeTPgxyJtiA2K9Oi9hjGhHQMABbWipV8OyJDriH28nIt_03aAuJ4s8tlJ3WeCJoEeZIzdaArnjjoCaw9KLZabPnxTDhtPY-Mr7yn9CHTeOdCh-WyMQFVFM_41bdPFrpfuyGxDBH6cfnqU8Pz9Es5WT0xdPG1prlWyA80vfCp1UZ2BsVIoldwfcN1Mr3CSrW3gSnt75TnOKlaXlFyZjOHIEba26QsxcVNujgCo238Y1GaS5e3ExxncbqtzippUuuoW3rNARceidVaB3S-6jTuXwQMZq2tkgCo1eF0W5_PWRdjUJ8vFdsj3--wUUSLmmW3Ta-yY29_odghQ1tKYoxeAyc7UkbNEnkrTTLPB7h9EC34_clivm6NUR0nnYqMIC_YQ7_Y5Sh7nsJ_A=w1352-h526-no)

### 2. Pixel và bộ lọc

#### 2.1. Bức ảnh giống như một hàm

Một bức ảnh bao gồm rất nhiều pixel. Một pixel có thể thể hiện cường độ của 3 màu: đỏ, xanh lục và xanh dương (R - Red, G - Green, B - Blue).

Một bức ảnh là một hàm <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{60}&space;\LARGE&space;f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{60}&space;\LARGE&space;f" title="\LARGE f" /></a> chạy từ <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{60}&space;\LARGE&space;R^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{60}&space;\LARGE&space;R^2" title="\LARGE R^2" /></a>  cho đến <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{60}&space;\LARGE&space;R^M" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{60}&space;\LARGE&space;R^M" title="\LARGE R^M" /></a>

- <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{60}&space;\LARGE&space;f(x,y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{60}&space;\LARGE&space;f(x,y)" title="\LARGE f(x,y)" /></a> trả về cường độ tại vị trí <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{60}&space;\LARGE&space;(x,y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{60}&space;\LARGE&space;(x,y)" title="\LARGE (x,y)" /></a>

- Được giới hạn trong một hình chữ nhật hữu hạn các cạnh a, b, c, d. Mỗi một pixel trong miền giới hạn đó có giá trị dao động từ 0 đến 255. <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{60}&space;\LARGE&space;f:[a,b][c,d]&space;\rightarrow&space;[0,&space;255]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{60}&space;\LARGE&space;f:[a,b][c,d]&space;\rightarrow&space;[0,&space;255]" title="\LARGE f:[a,b][c,d] \rightarrow [0, 255]" /></a>

<img src="https://lh3.googleusercontent.com/GK6o4mo7Tth1NXPYQwhA6mS_d4p9geoKLku6Gy0lHLBVWjqYVh3xFXQ8TvZzdgwup5NekkWWvHaNsDhz165zr9fMdfXuQdxwQTwHS5dd8-rYjUqCWJO-rI1giiZMWB3e3uEnPbOTq42Hbfci7EC2Im-bm1qenT53f6a6bcu_Wv_an1oJkjgyrszgxNn4Oyx5oEn-XeDi_jvcW0lTsMmG4GEQfI2ULPPCUwI20NJOW2iEVyjvs9NVJ0GRYEn7Yvky51rKcMxi-p-5Vt3M6NEKe1hFXYHSW0PuXjxEh8VM1VeFAR5dzX6quFpJzTnemkE4vXWLDpURgTFXJSO1cD68ImqSW-nRtEW5PCyXu2GLa2EidGgLf2J7t0SZukbusEbH3s07XNbzV3upwvxf8_sHwNnHHnlZ4lbG41E3Q3M0mzAWpYp2pzNCgnEeUAQE1tzxGS9X01OqJ2kLHCQL6J6F-SqCYz8PeTXiCx6_Qj2iTmSMbyqc8-yzZyMuIhhFP1KJSAIXho5CLI-vOSjm35_U_0iPKQwyQq_p-ERZ_azEFg21ifAi4EnCDxSV6euPjtUgj0ImaNNVqpvdrvA71AbNa_iEFzsq1WSVhp2eiu7ORk2vC1256Rq_6oM9M27MgfszOmjDlLYxhT8ltfDywahtV2RH1HoLAfMV_0ONoUsxfPAbcYW21ovo7saHMg=w946-h326-no">

#### 2.1. Biểu diễn ảnh bằng Ma trận

<img src="https://lh3.googleusercontent.com/cN5rIdJWOqTbFDbr3OJxxFLnt-X5wTJofR3OQwDolsHI1irFJBnOuBl-qDHGX7E3RGO-AHrqVO0BpxATslVYkn2MOQGM3yC3nzWexheRbbtLPjQdgXO9cqcYvd-TzXHkE0GFE_jHIGpEJxQwhgp2uFYuVmNB1FYO5TKcIqMBxnIsm7myl_yHqx6J9N8M8djICEnPX0tDrscTJmNiTjRPIpg_EdNw8iJJR6brfxkwK411AhKWVzNTC3rgy5V-FkDQoKRVhSM-RHraZxLTzGKOPCAv5yH1R4X2xtBjguf-odhioa0JAvWXhstZa5VDzemkBHmi5pPbyHDCfzC9EkoZr3bvVTTyfx18eLGzfGJqpXELyHBrweh6OnKBDhlZda_MNSjkgPAv9zSe22V6xjPNtknZFWCmb80t9Rg9SsRPqMqJp2juYa09nX9VUqmJUve4hCfUO_UvJZpR-ypUsjDuO1b1QB36QFX9JWrldA923q6L1sC5iXLINvsWktTwofeFJqxg-bm5ai2xpYh_fHAw46dmth_rHGMIkiCoq4PEvTG-z85Y6YkHMxSVJeUBOu4l8_n2Qxb6skocWexj5n9a8FTnWJR7LDBvs2TTH81I5HbdsTrfPDUlxJRY40lhABg=w960-h396-no">

<img src="https://lh3.googleusercontent.com/LNrlwYar7E0UT01oGAD60pf-iUgSAyYmFDYZfDVpwV36zuJ1yWHcmda3e2KPq0Wc5hcMw9wtQjJP-iukd8BXq73imIakjVTtecDT0A-too3unMaMAHP4T894J-2Ny64EADttnIDflwiW4gTf0EDpq4i9VpEP8_i7fRkNKZBe4cKWxWCyO_ZxyECjTRU0FtNO56LnX5yfNBUbLICAyfngE032_6tUz1O3wku1GFiaaT7cJzft6kHhlt0E4LAa9NVcDB9WuDu5xz2zvu1eTluN4O5WWUevv95W8sDutu2Up0uMqcnq1UTD-oVFOC6CWOnBxQR9cYMa2ab6nJ7v3yJ5p1zjr5LGKT-rNTLgC9WWdnHt0u0YNklX0hFRUfdw0KyHIlDc3pMjwOvb8WJ46PtVh1TLmJTfj0rsiqmYe3qimeQI2DpLaljSMCIPOM4D123nq173yGNx-1pTdMT8MXP8Y5JjIWa5m5Yz_6IymW2Gzobq56g2at3DOkgdAmrWhJX83cao7eMFmCrUAvbhVMjEiiO2AS4ZZrKq4CN7KN3qXDX-WRsUE7b9vR9F8pQfq78C7l4kuqtVINGEcB9u2mELL-GBvFKeLeQARrSNmZhZLAFRvv2GVjA2B3X04VJPM5Y=w992-h515-no">

Hình ảnh từ [Lecture 4 - CS131 - Stanford](http://vision.stanford.edu/teaching/cs131_fall1617/lectures/lecture4_pixels%20and%20filters_cs131_2016.pdf)

Các bạn có thể thấy mỗi pixel có thể là một hàm độc lập, và tập hợp các hàm độc lập này tạo nên một bức ảnh.

