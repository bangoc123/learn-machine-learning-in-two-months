

### 5. Bài toán phân loại (Classification)
----

   - **a. Mô hình Hồi quy Logistic (Logistic Regression Model)**
   
        <img src="https://lh3.googleusercontent.com/qyA4iDLK6VBQ8LBID8QM12f9NokoRRzf3J5SsOl1ALG8lrtdvpt4QixEQ7Xcnt0BfwIJLnA4WribRmIfOrHXfxM-Cmt2ugrs3nNdvh_7GSGW8sTMvPdmCi6nCzGxH1Sy5TReNV4kL--o7tSGHhdsXrtSzrmDCK0PG_orWhBJS4_hEmvE8tujI_S1fYtebKnWCu1VvU1DN8Opw8wFIOaj1bQZkucPgJcnx8JJer9Wc8sfRhzXRTus5ELxBKw0_-eLWzsnvVHlNK5u60_-f_E_5r24xBhDKe4YRrFiyrU0P2zUIrW0d_VK5WYzZkJlYuzBQjxWMdUnuGK30mn4-yn5iTYSPxdvFXZF8uhYcmbu2739EcF5lTfTCIbwtOz3KiHIdPr_yZYTgoEP-qcD9zWB7SRnAQwtQiwe3lWdLFVlfDTU9Z8pb8Fz56I1YRE4kGni_L08tXl9ThnuzVjOm3x8d__E42zZDIrjxVX1XtHcnKruvJhdxvfxoVshJtB5_Eod4Ms0UhfsYv8Gpd5K0GamXS3iWYaeTdx1FQ87JMOqKhFWEfr1qTER2IOJgM8phziFB2gUzZJWuFnINyHUQUPlf0Djht8shTDcykzrnZ34HNw0bWG5ErWfY3xzPAhfE_mVi9MenOMG7hl0quRn7h7WWyNroWAnXfwEhhGS4lE0gQdZaln0Q8HKTu3yxST6pZnCW_FKmzoVXAdLx1Mgkw=w900-h500-no" width=600>
        
        Như trên hình chúng ta có 2 bài toán phân loại: **Phân loại 2 lớp (Binary Classification)** và **Phân loại nhiều lớp (Multiclass Classification)**. 
        
      - **Phân loại 2 lớp (Binary Classification)**

           - Mô phỏng           
           <img src="https://lh3.googleusercontent.com/A67h2lsxXgkXLWeNDoC2b9IzL5iGFegRDtBPdulR8H1JVybn0dKXyJHW0DVvyw_AkuAjQ4gwhJibAcg4tKgoj2ABLB_bXl6WfDFWkP8-3UXTmAdUYvmc7ug6DwO37_VoeKhbgwHYxqh9NNuKQbsF0l_DSMC4HwmZ4bEJEjSkDqMhiDxkbB6a0AYr51UzB49UUYDgviIfptVu9tGJV8sKxF7RcP9ByO_C9N5fK6lBSWmgQz00w30IJVDFL_1Fow47JjSiV8midSh_ptpqB54lnF2tFSidOAwUkGsiFhQYeXZSWiHF1iFpw01O5jRbsnJNlpKlR6L8EG3gNHZITDSNkoZRph2iZj4fsP9uyEWpEvBSRD0GCckZZxFqMM7YkrVi6kZclJwK8YQI0pnMiAAQy2LtAVMKitw-qusHU9DCxyJgpuXU7asYd879LXb8vEWnBV2_ouvPibinkxSpJMAN3bHz-9lMcDysWdGovcOQuVP7iXeoHmcTC9F7BBxlL-s2zqNSKy7VdY8Vw-XBbkWYz70Nq3ZX7xrjmcaXPaMiNsJF4p0SFayzoGfI1zaoOhhEApqdc5zM6fkaTfuOmfAjUZQUN88aqQOEHLyC8CHyk-8odzOqbAfldUxTTap79XSJVzAFm-ghWJA7Hm_y_oO9uxUGy8dv9UMmqGCJ-EEF9-QkoKucNO9wvVSQJtsQmsUiGKW8B6ZdwwwYG2wWaw=s800-no" width=400>

           Bài toán ở đây rất đơn giản, cho 1 dữ liệu bất kỳ có giá trị dựa theo trục <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X_1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;X_1" title="X_1" /></a> và <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X_2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;X_2" title="X_2" /></a>. Làm thế nào ta xác định được điểm đó là thuộc lớp **ô tròn màu đỏ** hay **ô vuông màu xanh**. Vậy chúng ta có thể kết luận đây và bài toán phân loại.
        
      - **Phân loại nhiều lớp (Multiclass Classification)**

           - Mô phỏng
           
           <img src="https://lh3.googleusercontent.com/7yNQ_CW4yCcDRKi8lT73ImAcsH-xqR6Ew4CGCeEyfwwY5960y557GeN7mWatO8h-6P8TS0qWd3BarekpYscu0kXJkL3_wb8foAba5VQWXVHhaPQ1V6Mh85oBciQwsPcHt1fhikqzWEUzIP8F2HCOJ9SWyki43vuWwUcSoAtFXhckncUHq0bTsiSU_Qz7W-Epc62h2KECf4mGErtDyk1oXMfeQH2UIP19F4gchagbrG7t99WH94GmoCNstfPg_MmH9CycHsBZHN9mXwrpaWKf1adjA0FzMT4i8SVSZ8bjkW29YHMZLA3dSypNhDN38NwjEqPYgsRSTKf5EWQhT9yBUJ5wgyrKYIiQSxzmSLCuUpVrcVZ7qgQZEiRIziVs7DOonCqZ0qaBZiIoDhkfeiyWFFw2sPQmoSJ-7fuKmK_8tywJjxhFU1fJKwX0e3amVPSqStjW45BmIw_nFi_vF3YSJOcfOnJFX2SJmZMx-5A9aIGCvMVy407F9qXTSuiMjmoyuPRxbaFPK-1TAlddUR0-o6LoUzhIHknupvcThrCnpHNMsCOaqH9FkmuruG-3hZyMhMIs9DsdoWMJcLsO-LBrsJPdQNa2xawxMd_z0GUh0z3v2cz5S9709k54Y4Cl9XGxOZMjQ1CZkgLJ4P_jUKUoVvl6vh-GEepVSHoF5Bflz37sCbAVqam4Th-Ycj3qjBoN98uJtaLLAuze67cgSw=s800-no" width=400>
           
           Tương tự bài toán phân loại 2 lớp, là đầu ra chúng ta cần một điểm thuộc 1 lớp nhất định và điều khác biệt ở đây đó là số lượng lớp luôn lớn hơn 2. Ở trong ví dụ này là 3 lớp. Các model phân biệt ảnh lớn có thể có đến 1000 lớp như [ImageNet](http://www.image-net.org/).

     - **Cách tiếp cận bài toán này**
      
         Tôi sẽ lấy bài phân loại 2 lớp để các bạn hình dung trước. Có 2 cách tiếp cận để giải bài toán này:

         | Discriminative model	 | Generative model|
         | ------ | ------ |
         | Tìm đường ngăn cách (Decision Boundary)	 | Tìm phân bố xác suất của dữ liệu (Probability distributions of the data)|  
         | <img src="https://lh3.googleusercontent.com/7DTs01OsbYlxFWjYgpva10MuVbgYBp3-wONSrDB-TM86w041bW4fAazswU4AQui4qKlDhtGQw0fFtFZGT6bPwzWrtAH85eYbszq9X0q-53vgV8AGbn5y3QJGar53_35GaZ9EuPU23oyxDCzFp2z3mQ_rd_bORqs4WHOtC9U3cqqE2I262ib7sy1XkBQoDI1vIry0Cs9SYA8Ka3YrOK6OTKo6ZYjbqlk0uCspkze9sx4o_7sc5dJjw90GMhQ_JFWFtVAZxm1qFuDUx7V3rFxly1d7QCvhZir_MwXfHy97rE-nKCGA_3wbbQnoM9WOIHkZER3UBH6h4xMMGqwHF4M96v1JibVh_01eBnttTYySfIymOJV8yK8m3qyiqhtH4WxHqx9tUjvgVxI8QEz9ET56dDDGkx_Sb24raSdOmB6IOuPj255n-u9CiJEI6qV9jPK9KXWiaLL1RQv0bVPvWpyTPexDNqUQQy1Y1xGKt3IKf0IJMft0yCM8Wo6oTGRNiLlglaakih2H7J7FxJvgHvq191gdTm_GMek0eSl8HuuZmzUpOXV_a7EkjFFqLdhdr2UyHdjnxobioc_t48l2QElq0NS3SXyW5nbLtpU4dTvpJ7pi_kthEsmc4m2fgBEbsA7V9bNmp7s54r5IcjtauTohNPUKzEvRqLysm39IR-DImKjINlI93CZ48MYTn9E2rlkuzlU6SS5rkAwwSxQhDw=w500-h300-no">	 | <img src="https://lh3.googleusercontent.com/cnzJHljtpTIop68rQiiMx3Y4Oel7g3tHDUmsdvvG-Ca_BWXp6a9Swo7aqdMskYgsX4puyTT0Kb3imvVzdpqLUQ_vyJ1EqJwRdSQwl-MU1QfNqoYxeTFhsluiucaOFsH4m3wfkQdkMqA1hXzrsoOxLkIWN5Nf88Z6l1cVghE-oSa8_olTY1qsq0rJPjjXHZNWHVO9a5QMHb5Uh7KPlayNpiAIMRtFDC1xRuYgctuFodndZGd-GJWyI-qwtJWZoCMkTM0eNJhDXlXJtXMMURwwSCP3vxHPhX2Vg-AqeG-_9BKgvQyQ8_bz342qVDcWyyOFtfoxF9nhHufCacuAyHgMBKAPs543bwQo6aDWSYEtWVPWA-oNSOjtJY7p7LIAFCoKTWmW3fVZjp-VMFyiQo2gIlLDMsZsC2htYezxQGb6Wk9PSmifZYHUzVQokZmi2kXk-i84hyVOoONxNusSZtBrMvZcC5oeC9_IL5aINkpw_oAb-jW1nw8drdNffCtAU614namIbFTzoiwe6TxkNoXtAfARKsCzKmsVuBTfJthRw92emSSZrUH9ehnkAbhjCUuFw_3Ozf91e_ghXSH26kASnh-GaKE172yS2TgwNZvxprJXgGicBzNEGLYhqrxyABNnpyRGLZ7ZB9Vea2dVRfnDLw6W0pmaJbLu_byVYl41t63PUMDosNrJIJ-J5DgABFNgGjZEpJQflFKGgUYG_g=w500-h300-no">|  

         Theo [Supervised Learning cheatsheet](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning#linear-models)
         
      - **Tiếp cận theo hướng Discriminative**
		Nhắc lại chúng ta sẽ tìm một đường ngăn cách để có thể phân biệt 2 vùng data với trường hợp phân loại 2 lớp hoặc nhiều vùng data trong trường hợp phân loại nhiều lớp.

		Trong trường hợp phân loại 2 lớp:
		
		<img src="https://lh3.googleusercontent.com/A67h2lsxXgkXLWeNDoC2b9IzL5iGFegRDtBPdulR8H1JVybn0dKXyJHW0DVvyw_AkuAjQ4gwhJibAcg4tKgoj2ABLB_bXl6WfDFWkP8-3UXTmAdUYvmc7ug6DwO37_VoeKhbgwHYxqh9NNuKQbsF0l_DSMC4HwmZ4bEJEjSkDqMhiDxkbB6a0AYr51UzB49UUYDgviIfptVu9tGJV8sKxF7RcP9ByO_C9N5fK6lBSWmgQz00w30IJVDFL_1Fow47JjSiV8midSh_ptpqB54lnF2tFSidOAwUkGsiFhQYeXZSWiHF1iFpw01O5jRbsnJNlpKlR6L8EG3gNHZITDSNkoZRph2iZj4fsP9uyEWpEvBSRD0GCckZZxFqMM7YkrVi6kZclJwK8YQI0pnMiAAQy2LtAVMKitw-qusHU9DCxyJgpuXU7asYd879LXb8vEWnBV2_ouvPibinkxSpJMAN3bHz-9lMcDysWdGovcOQuVP7iXeoHmcTC9F7BBxlL-s2zqNSKy7VdY8Vw-XBbkWYz70Nq3ZX7xrjmcaXPaMiNsJF4p0SFayzoGfI1zaoOhhEApqdc5zM6fkaTfuOmfAjUZQUN88aqQOEHLyC8CHyk-8odzOqbAfldUxTTap79XSJVzAFm-ghWJA7Hm_y_oO9uxUGy8dv9UMmqGCJ-EEF9-QkoKucNO9wvVSQJtsQmsUiGKW8B6ZdwwwYG2wWaw=s800-no" width=400>
	
		Phân bố xác suất:
		
		Xác suất 1 điểm là màu xanh:
	
		<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;P(y_i=1|x_i;&space;\theta)&space;=&space;h_{\theta}(x_i)&space;=&space;g(\theta^Tx_i)&space;\&space;\&space;\&space;(1)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;P(y_i=1|x_i;&space;\theta)&space;=&space;h_{\theta}(x_i)&space;=&space;g(\theta^Tx_i)&space;\&space;\&space;\&space;(1)" title="\large P(y_i=1|x_i; \theta) = h_{\theta}(x_i) = g(\theta^Tx_i) \ \ \ (1)" /></a>

		Xác suất 1 điểm là màu đỏ: 

		<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;P(y_i=0|x_i;&space;\theta)&space;=&space;1&space;-&space;h_{\theta}(x_i)&space;=&space;1-&space;g(\theta^Tx_i)&space;\&space;\&space;\&space;(2)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;P(y_i=0|x_i;&space;\theta)&space;=&space;1&space;-&space;h_{\theta}(x_i)&space;=&space;1-&space;g(\theta^Tx_i)&space;\&space;\&space;\&space;(2)" title="\large P(y_i=0|x_i; \theta) = 1 - h_{\theta}(x_i) = 1- g(\theta^Tx_i) \ \ \ (2)" /></a>

        Viết gọn lại 2 công thức bên trên ta sẽ được:

        <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;P(y_i|x_i;&space;\theta)&space;=&space;h_{\theta}(x_i)^{y_i}&space;(1-h_{\theta}(x_i))^{1-y_i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;P(y_i|x_i;&space;\theta)&space;=&space;h_{\theta}(x_i)^{y_i}&space;(1-h_{\theta}(x_i))^{1-y_i}" title="\large P(y_i|x_i; \theta) = h_{\theta}(x_i)^{y_i} (1-h_{\theta}(x_i))^{1-y_i}" /></a>

        - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;y_i&space;=&space;1&space;\rightarrow&space;P(y_i=1|x_i;&space;\theta)&space;=&space;h_{\theta}(x_i)^{1}&space;(1-h_{\theta}(x_i))^{1-1}&space;=&space;h_{\theta}(x_i)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;y_i&space;=&space;1&space;\rightarrow&space;P(y_i=1|x_i;&space;\theta)&space;=&space;h_{\theta}(x_i)^{1}&space;(1-h_{\theta}(x_i))^{1-1}&space;=&space;h_{\theta}(x_i)" title="\large y_i = 1 \rightarrow P(y_i=1|x_i; \theta) = h_{\theta}(x_i)^{1} (1-h_{\theta}(x_i))^{1-1} = h_{\theta}(x_i)" /></a>

        - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;y_i&space;=&space;0&space;\rightarrow&space;P(y_i=0|x_i;&space;\theta)&space;=&space;h_{\theta}(x_i)^{0}&space;(1-h_{\theta}(x_i))^{1-0}&space;=1-h_{\theta}(x_i)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;y_i&space;=&space;0&space;\rightarrow&space;P(y_i=0|x_i;&space;\theta)&space;=&space;h_{\theta}(x_i)^{0}&space;(1-h_{\theta}(x_i))^{1-0}&space;=1-h_{\theta}(x_i)" title="\large y_i = 0 \rightarrow P(y_i=0|x_i; \theta) = h_{\theta}(x_i)^{0} (1-h_{\theta}(x_i))^{1-0} =1-h_{\theta}(x_i)" /></a>

        Đoạn này bạn có thấy giống [Phân phối Bernoulli](https://vi.wikipedia.org/wiki/Ph%C3%A2n_ph%E1%BB%91i_Bernoulli) không?
        
        Chú ý rằng đây mới là xác suất của **một điểm** sẽ là xanh hay đỏ. Bài toán của chúng ta cần xét trên toàn bộ tập **data set**.

        <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;X&space;=&space;[x_1,&space;x_2,&space;...,&space;x_m]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;X&space;=&space;[x_1,&space;x_2,&space;...,&space;x_m]" title="\large X = [x_1, x_2, ..., x_m]" /></a> với <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;X&space;\in&space;\mathbb{R}^{d\times&space;m}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;X&space;\in&space;\mathbb{R}^{d\times&space;m}" title="\large X \in \mathbb{R}^{d\times m}" /></a> và <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;Y&space;=&space;\left&space;\{&space;0,&space;1&space;\right&space;\}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;Y&space;=&space;\left&space;\{&space;0,&space;1&space;\right&space;\}" title="\large Y = \left \{ 0, 1 \right \}" /></a>

        Vì <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;x_1,&space;x_2,&space;...,&space;x_m" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;x_1,&space;x_2,&space;...,&space;x_m" title="\large x_1, x_2, ..., x_m" /></a> là các biến độc lập nhau cho nên:

        <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;P(Y|X;&space;\theta)&space;=&space;P(y_1|x_1;&space;\theta)&space;P(y_2|x_2;&space;\theta)&space;...&space;P(y_m|x_m;&space;\theta)&space;=&space;\prod_{i}^{m}&space;P(y_i|x_i;&space;\theta)&space;=&space;\prod_{i}^{m}&space;h_{\theta}(x_i)^{y_i}&space;(1-h_{\theta}(x_i))^{1-y_i}&space;\&space;\&space;\&space;(4)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;P(Y|X;&space;\theta)&space;=&space;P(y_1|x_1;&space;\theta)&space;P(y_2|x_2;&space;\theta)&space;...&space;P(y_m|x_m;&space;\theta)&space;=&space;\prod_{i}^{m}&space;P(y_i|x_i;&space;\theta)&space;=&space;\prod_{i}^{m}&space;h_{\theta}(x_i)^{y_i}&space;(1-h_{\theta}(x_i))^{1-y_i}&space;\&space;\&space;\&space;(4)" title="\large P(Y|X; \theta) = P(y_1|x_1; \theta) P(y_2|x_2; \theta) ... P(y_m|x_m; \theta) = \prod_{i}^{m} P(y_i|x_i; \theta) = \prod_{i}^{m} h_{\theta}(x_i)^{y_i} (1-h_{\theta}(x_i))^{1-y_i} \ \ \ (4)" /></a>

        Đọc thêm về [Biến cố độc lập](https://vi.wikipedia.org/wiki/%C4%90%E1%BB%99c_l%E1%BA%ADp_th%E1%BB%91ng_k%C3%AA) để hiểu tại sao lại sử dụng tích này.        

        **Hàm sigmoid**

        Ở đây ta sử dụng hàm sigmoid: 

        <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;h_{\theta}(x)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-z}}&space;=&space;\frac{1}{1&space;&plus;&space;e^{-{\theta}^Tx}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;h_{\theta}(x)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-z}}&space;=&space;\frac{1}{1&space;&plus;&space;e^{-{\theta}^Tx}}" title="\large h_{\theta}(x) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-{\theta}^Tx}}" /></a>

        Trong đó:

        <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;z&space;=&space;\theta^Tx" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;z&space;=&space;\theta^Tx" title="\large z = \theta^Tx" /></a>

        Kết hợp với hàm số (4) ta có:

        <a href="https://www.codecogs.com/eqnedit.php?latex=P(Y|X;&space;\theta)&space;=&space;\prod_{i}^{m}&space;h_{\theta}(x_i)^{y_i}&space;(1-h_{\theta}(x_i))^{1-y_i}&space;\&space;\&space;\&space;=&space;\prod_{i}^{m}&space;(\frac{1}{1&space;&plus;&space;e^{-{\theta}^Tx_i}})^{y_i}&space;(1-\frac{1}{1&space;&plus;&space;e^{-{\theta}^Tx_i}})^{1-y_i}&space;\&space;\&space;\&space;(5)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P(Y|X;&space;\theta)&space;=&space;\prod_{i}^{m}&space;h_{\theta}(x_i)^{y_i}&space;(1-h_{\theta}(x_i))^{1-y_i}&space;\&space;\&space;\&space;=&space;\prod_{i}^{m}&space;(\frac{1}{1&space;&plus;&space;e^{-{\theta}^Tx_i}})^{y_i}&space;(1-\frac{1}{1&space;&plus;&space;e^{-{\theta}^Tx_i}})^{1-y_i}&space;\&space;\&space;\&space;(5)" title="P(Y|X; \theta) = \prod_{i}^{m} h_{\theta}(x_i)^{y_i} (1-h_{\theta}(x_i))^{1-y_i} \ \ \ = \prod_{i}^{m} (\frac{1}{1 + e^{-{\theta}^Tx_i}})^{y_i} (1-\frac{1}{1 + e^{-{\theta}^Tx_i}})^{1-y_i} \ \ \ (5)" /></a>
		
		Vậy tại sao phải sử dụng hàm này: Bạn chú rằng đầu ra của bài toán này chỉ là 2 giá trị **0** và **1** mà không phải là vô số giá trị thuộc tập số thực giống như [bài toán hồi quy tuyến tính]([https://github.com/bangoc123/learn-machine-learning-in-two-months/tree/master/models/linear-regression](https://github.com/bangoc123/learn-machine-learning-in-two-months/tree/master/models/linear-regression)).
		
		Hàm sigmoid sẽ giúp đầu ra của biểu thức <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;h_{\theta}(x_i)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;h_{\theta}(x_i)" title="\large h_{\theta}(x_i)" /></a> giới hạn trong khoảng **[0,1]**, Việc chuyển đổi này thoả mãn nhu cầu cần tìm ra xác suất của một điểm bất kỳ là xanh hay đỏ ở công thức (1) và (2) ( giá trị này nhỏ nhất bằng **0** và lớn nhất bằng **1**) :
		
		<img src="https://lh3.googleusercontent.com/wggFdojRXGH5q_ImaM4dKYnKIEyHwmN-go1jNk1SCqqlyg-SAxITfU3ODVyO_a3I7UXqxtyUqisGfppR71LVn2JduX3-yTnL67cLOliCEpXJ1tMXmqzx0urQfS5BbVEmVh1H_KB3Gc3IPZ0j3LMsnLIGEQcyKvcgbbI2nGxXzkgqtkL7j6qpvK_-fVnilgaZxZ5d2kKPGqcI1v04yxe7FYhY6HylJOwB0IsEEdppKVwcLmUAzWrHHXZ-xGUYqjjKgOEP8EyXgLTIEnyXd7Xw00lhNK4z2IZRZE8z9biL9yVCR7wh8I6a8ygENuzM7cIBbi1wWowu84Q3fjDIieElNn8wTawYTg-4UAbT89OQ567f71DM-Bt5AhEAKrCpt3rht5O69io0u4gvM4nrRejzyWOj3pycaUgVRayTucLePidpbfq0ZhLKrXdDwjGGIqs4oCWZqzvtyIt-dOBw3W9ponEgzuXmwSx5uATctBZQjD1kEmG6FiGEzhuMBMAd65nLQQfn--5czUNZtewptqXf1AeRaHGcnzzYzr2MZXZmjM7rDjoPoGPoMgObkhyqAfbATgVsWguiAt0LnIpf2Y3CTm7v1tXAX00IKYf9k64iFLyqeuJYL6b1vVvRWthXb7jGQLfWaa2OACAQ7vED4E21AFK0-rCaCq9sV_X9nNQCqT2uC1r_isqjH5w1_osuOlN5EN5yXWelmB9r75qqqvqACXAT=w320-h213-no" />
	
		Nhìn vào đồ thị chúng ta có thể thấy, dù  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\LARGE&space;z" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\LARGE&space;z" title="\LARGE z" /></a> có lớn thế nào thì đầu ra của hàm tối đa là **1**, cũng như kết quả của <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\LARGE&space;z" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\LARGE&space;z" title="\LARGE z" /></a> có nhỏ cỡ nào thì đầu ra của hàm tối đa là **0**.

        **Toàn bộ quy trình học từ một cặp dữ liệu**. Quá trình này được biểu diễn dưới một mạng neuron đơn giản.

        - Bước 1: Nhân vector. <img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;x^i" title="x^i" /> là một điểm dữ liệu có chiều (nx1). <img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;\theta" title="\theta" /> có chiều (nx1). Sau này đến bài mạng neuron bạn sẽ học kỹ hơn về chiều của tham số.
        - Bước 2: Phép biến đổi phi tuyến, chuyển từ giá trị bất kỳ z về một giá trị nhỏ hơn 1 và lớn hơn 0 thông qua hàm **sigmoid**.
        - Từ xác suất này kết luận là điểm dữ liệu này thuộc lớp 0 hay lớp 1. Giải thích ký hiệu: <img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;1\left&space;\{&space;True&space;\right&space;\}&space;=&space;1" title="1\left \{ True \right \} = 1" /> và <img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;1\left&space;\{&space;False&space;\right&space;\}&space;=&space;0" title="1\left \{ False \right \} = 0" />. Trong trường hợp này đầu ra của hàm sigmoid lớn hơn một threshold cho trước (ở đây ta đặt là 0.5) thì sẽ kết luận điểm đó **thuộc lớp 1** và ngược lại.

        <img src="../../images/Logistic .jpg">

        **Loss function**

        Rõ ràng chúng ta thấy để model chính xác thì chúng ta phải **cực đại hoá** hàm likelihood <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;P(Y|X;&space;\theta)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\small&space;P(Y|X;&space;\theta)" title="\small P(Y|X; \theta)" /></a>. Hàm này thể hiện xác suất đồng thời xảy tất cả các điểm dữ liệu, càng nhiều điểm dữ liệu xuất hiện, mô hình càng chính xác.

        Xem thêm về [Cực đại hoá Likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).

        <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;max&space;\&space;P(Y|X;&space;\theta)&space;\Leftrightarrow&space;max&space;\&space;log&space;P(Y|X;&space;\theta)&space;\Leftrightarrow&space;max&space;\&space;log(\prod_{i}^{m}&space;h_{\theta}(x_i)^{y_i}&space;(1-h_{\theta}(x_i))^{1-y_i})&space;\Leftrightarrow&space;max&space;\&space;\sum_{i}^{m}&space;y_i&space;log(h_\theta(x_i))&space;&plus;&space;(1-y_i)log(1-h_\theta(x_i))&space;\Leftrightarrow&space;min&space;\&space;\frac{-1}{m}&space;\sum_{i}^{m}&space;y_i&space;log(h_\theta(x_i))&space;&plus;&space;(1-y_i)log(1-h_\theta(x_i))&space;\Leftrightarrow&space;min&space;\&space;J(\theta)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\small&space;max&space;\&space;P(Y|X;&space;\theta)&space;\Leftrightarrow&space;max&space;\&space;log&space;P(Y|X;&space;\theta)&space;\Leftrightarrow&space;max&space;\&space;log(\prod_{i}^{m}&space;h_{\theta}(x_i)^{y_i}&space;(1-h_{\theta}(x_i))^{1-y_i})&space;\Leftrightarrow&space;max&space;\&space;\sum_{i}^{m}&space;y_i&space;log(h_\theta(x_i))&space;&plus;&space;(1-y_i)log(1-h_\theta(x_i))&space;\Leftrightarrow&space;min&space;\&space;\frac{-1}{m}&space;\sum_{i}^{m}&space;y_i&space;log(h_\theta(x_i))&space;&plus;&space;(1-y_i)log(1-h_\theta(x_i))&space;\Leftrightarrow&space;min&space;\&space;J(\theta)" title="\small max \ P(Y|X; \theta) \Leftrightarrow max \ log P(Y|X; \theta) \Leftrightarrow max \ log(\prod_{i}^{m} h_{\theta}(x_i)^{y_i} (1-h_{\theta}(x_i))^{1-y_i}) \Leftrightarrow max \ \sum_{i}^{m} y_i log(h_\theta(x_i)) + (1-y_i)log(1-h_\theta(x_i)) \Leftrightarrow min \ \frac{-1}{m} \sum_{i}^{m} y_i log(h_\theta(x_i)) + (1-y_i)log(1-h_\theta(x_i)) \Leftrightarrow min \ J(\theta)" /></a>

        Chú ý ta cần lấy log để tránh trường hợp khi số lượng điểm dữ liệu quá lớn dẫn đến tích của nhiều số nhỏ hơn 0 sẽ tiệm cận về không. Vì thế thay vì sử dụng phép tích, chúng ta dùng hàm log để việc cực đại **tích này** thành cực đại **một tổng**.

        Vì vậy:

        Hàm Loss Function của chúng ta chính là:

        <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;J(\theta)&space;=&space;\&space;\frac{-1}{m}&space;\sum_{i}^{m}&space;y_i&space;log(h_\theta(x_i))&space;&plus;&space;(1-y_i)log(1-h_\theta(x_i))" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\small&space;J(\theta)&space;=&space;\&space;\frac{-1}{m}&space;\sum_{i}^{m}&space;y_i&space;log(h_\theta(x_i))&space;&plus;&space;(1-y_i)log(1-h_\theta(x_i))" title="\small J(\theta) = \ \frac{-1}{m} \sum_{i}^{m} y_i log(h_\theta(x_i)) + (1-y_i)log(1-h_\theta(x_i))" /></a>

          

          **Tối ưu Loss Function**

          Tương tự bài toán Linear Regress, chúng ta sẽ sử dụng thuật toán **Gradient Descent**.

          Cập nhật <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a> sao cho hàm Loss nhỏ nhất có thể:

          <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta_1&space;=&space;\theta_0&space;-&space;\alpha&space;\bigtriangledown&space;J(\theta_0)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta_1&space;=&space;\theta_0&space;-&space;\alpha&space;\bigtriangledown&space;J(\theta_0)" title="\large \theta_1 = \theta_0 - \alpha \bigtriangledown J(\theta_0)" /></a>

          Công thức tính đạo hàm riêng của các phần tử trong <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\theta" title="\large \theta" /></a>:

          <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;J(\theta)}{\partial&space;\theta_i}&space;=&space;\frac{1}{m}&space;\sum_{i=1}^{m}(h_\theta(x^i)&space;-&space;y^i)x_j^i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;J(\theta)}{\partial&space;\theta_i}&space;=&space;\frac{1}{m}&space;\sum_{i=1}^{m}(h_\theta(x^i)&space;-&space;y^i)x_j^i" title="\frac{\partial J(\theta)}{\partial \theta_i} = \frac{1}{m} \sum_{i=1}^{m}(h_\theta(x^i) - y^i)x_j^i" /></a>

          Vậy câu hỏi đặt ra tiếp theo rằng nếu chỉ sử dụng hàm **sigmoid** có thể đúng xử lý bài toán phân loại nhiều lớp không?


   - **b. Tiếp cận bài toán phân loại nhiều lớp**

        Để trả lời câu hỏi bên trên ta có thể vẽ lại mô hình phân loại:

        <img src="../../images/logmu.png">


        Các bạn đừng sợ khi nhìn vào ảnh này. Đây chỉ là cách biểu thị các phép tính toán.

        Ở bước 1, việc các mũi tên trỏ vào <a href="https://www.codecogs.com/eqnedit.php?latex=z_1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?z_1" title="z_1" /></a> thể  hiện phép nhân <a href="https://www.codecogs.com/eqnedit.php?latex=z_1&space;=&space;\theta_1^Tx" target="_blank"><img src="https://latex.codecogs.com/svg.latex?z_1&space;=&space;\theta_1^Tx" title="z_1 = \theta_1^Tx" /></a>. 
        
        Bản chất mô hình này có thể coi là một Neural Network với 1 lớp ẩn. Vấn đề này sẽ được đề cập đến trong bài tiếp theo.

        Có thể nhận ra rằng mục tiêu của chúng ta là phân phối được tạo ra khi đi qua hàm **signmoid** phải thoả mãn tổng bằng 1. Nhưng điều này không được đảm bảo.

        Vì thế khả năng dự đoán của mô hình không còn chính xác. Cho nên, ta cần một thuật toán có tên **Softmax Regression**.

        Công thức của hàm Softmax:

        <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma_j&space;=&space;\frac{e^{z_j}}{\sum_{j}^{K}e^z_j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\sigma_j&space;=&space;\frac{e^{z_j}}{\sum_{j}^{K}e^z_j}" title="\sigma_j = \frac{e^{z_j}}{\sum_{j}^{K}e^z_j}" /></a>

        Trong đó **K** là số lớp.

        Áp dụng vào trường hợp bên trên ta có:

        <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma_j&space;=&space;\frac{e^{z_j}}{\sum_{j}^{3}e^z_j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\sigma_j&space;=&space;\frac{e^{z_j}}{\sum_{j}^{3}e^z_j}" title="\sigma_j = \frac{e^{z_j}}{\sum_{j}^{3}e^z_j}" /></a>

        Ta có biểu đồ mới:

        <img src="../../images/softmax.png">

        Rõ ràng ta có thể thấy rằng hàm Softmax đã chuyển phân phối của <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;z" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;z" title="\large z" /></a> về phân phối của <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\sigma" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;\sigma" title="\large \sigma" /></a> sao cho thoả mãn điều kiện là tổng các <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\sigma" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;\sigma" title="\large \sigma" /></a> bằng 1.

        Và dĩ nhiên rằng giá trị <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\sigma" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;\sigma" title="\large \sigma" /></a> lớn nhất sẽ đại diện cho lớp mà ta dự đoán ra.

        Lúc này hàm Loss Function sẽ có dạng:

        <a href="https://www.codecogs.com/eqnedit.php?latex=J(\theta)&space;=&space;-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}(1\left&space;\{&space;y^i&space;=&space;k&space;\right&space;\}log(\sigma_k))" target="_blank"><img src="https://latex.codecogs.com/svg.latex?J(\theta)&space;=&space;-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}(1\left&space;\{&space;y^i&space;=&space;k&space;\right&space;\}log(\sigma_k))" title="J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}(1\left \{ y^i = k \right \}log(\sigma_k))" /></a>

        Nếu bạn thay K bằng 2 vào công thức này sẽ được chính hàm mất mát của bài toán phân loại 2 lớp.
- **c. Mô phỏng hoá quá trình Training**

    <img src="../../images/logis.gif">

    Quan sát ảnh động này ta sẽ thấy quá trình training sẽ diễn ra thế nào, lúc đầu khi khởi tạo <a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /></a> độ sai lệch sẽ rất lớn. Nhưng khi cập nhật <a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /></a> thì đường thẳng <a href="https://www.codecogs.com/eqnedit.php?latex=h_\theta(x)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?h_\theta(x)" title="h_\theta(x)" /></a> đã có khả năng phân biệt giữa 2 lớp.


    Thử nghiệm [tại đây](https://playground.tensorflow.org/#activation=tanh&regularization=L1&batchSize=19&dataset=gauss&regDataset=reg-plane&learningRate=0.01&regularizationRate=0&noise=30&networkShape=3&seed=0.49475&showTestData=false&discretize=false&percTrainData=70&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)








