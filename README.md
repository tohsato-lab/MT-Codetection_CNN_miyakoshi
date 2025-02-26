# Mean Teacher + Co-detection CNN
# 概要
自分の作成したコードを動かすための説明をします．  
まずクローンした後に足りないファイルを作成します．

* data (特に使用していない)
* log (tensorboardのlogを保存するファイル)
* result (推論画像を個体名で保存するファイル)
* result2 (推論画像を連番で保存するファイル)
* utils (dataloaderやdata拡張などが入るファイル)
* weight (学習の重みを保存するファイル)
* predict.py (推論を行うコード)
* train.py (学習を行うコード)
* log_U-Net.csv (logを保存するcsv)
* result.csv (推論結果のIoUを保存するcsv)

# 学習
学習を行う際は  
`python train.py`    
を実行すると学習を行うと思います．

# 推論
推論を行う際は  
`python predict.py`    
を実行すると推論を行うと思います．

# tensorboardで学習経過を確認
tensorboardで経過を確認する場合は  
`tensorboard --logdir log`    
を実行すると見れると思います．

# 使用したパッケージ
* python = 3.10.14
* pytorch (CUDA 12.4)
* pandas = 2.2.1
* tqdm = 4.66.4
* tensorboard = 2.12.1



