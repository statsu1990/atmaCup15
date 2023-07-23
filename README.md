# atmaCup 5th Place Solution
atmaCupおつかれさまでした。
運営の皆様、楽しいコンペを開催していただいてありがとうございました。

## コンペ概要
https://www.guruguru.science/competitions/21/

### タスク
- user_id, anime_idが与えられ、アニメ作品のユーザー評価を予測する。
- あるユーザーがアニメを視聴したがスコアを付けなかったケースについてユーザーが付けるであろうスコアを知りたい、というモチベーション。

### データ
- 学習データには、user_id, anime_id, score, anime_id毎の情報がある。
- 予測対象データ(テストデータ)には、user_id, anime_id, anime_id毎の情報がある。user_idの一部は学習データに含まれていない。
- テストデータはscoreそのものはないが、「user_idがanime_idを視聴した」という視聴情報を含んでいる(運営から明示的にどこかで説明されていたかは未確認)。

## Solution
### Over View
- Score
  - CV 1.1582, Public LB 1.1754, Private LB 1.1409
- seen user, unseen userの扱い方
  - seenとunseenでモデルを分ける。具体的には学習データの分割方法(CV)を変えてモデルを作成する。
- CV
  - seen user model
    - KFold。5 fold。
  - unseen user model
    - GroupKFoldでtrain_set, unseen_test_setに分割。train_setをGroupShuffleSplitでseen_train_set, unseen_train_setに分割。5 fold。
- 特徴量
  - Proneによる視聴グラフのEmbedding特徴量
  - Score情報を使った特徴量
- 学習モデル
  - xgboostでの回帰。seen user model 5 foldの平均、unseen user model 5 foldの平均。
- 後処理
  - clip(1, 10)

### seen user, unseen userの扱い方
方針: seen userとunseen userで効く特徴量が異なりそうなことからseenとunseenでモデルを分けた。

詳細: 
- 学習: seenとunseenで学習データを分けて別々にモデルを学習する。
  - 学習データの分割方法(CV)を変える。
  - 特徴量の種類や計算方法は同じ。Scoreを使った特徴量は値が変わる。
  - 学習モデルは同じ。
- テスト: seen userにはseen model、useen userにはunseen modelを使う。

### CV
方針: seen user model, unseen user modelの用途に合った分割方法とする。

詳細: 
- 学習データの分割方法
  - seen user model学習・評価用
    - KFoldでtrain_data_set, val_data_setに分割。
    - 5 fold
  - unseen user model学習・評価用
    - GroupKFoldでtrain_data_set, unseen_val_data_setに分割。train_data_setをGroupShuffleSplitでseen_train_data_set, unseen_train_data_setに分割(2分割)。
    - unseen_train_data_setのscoreを削除(特徴量作成時にscoreを使わない。学習の正解データとしては使う。)
    - 5 fold
- CV scoreの計算方法
  - 0.73 * seen user model rmse (5 fold average) + 0.23 * unseen user model rmse (5 fold average)

### 特徴量
方針: 
- seen userに効きそうなので、リークしない範囲でScoreを絡めてエンコーディングする。
- unseen userに効きそうなので、視聴情報を頑張って取り込む。
- 作業時間がないのでseenとunseen modelで特徴量の計算方法を分けない。(破綻しないように注意して設計する)

詳細:
- 特徴量の計算に使うデータセット
  - 視聴情報に関する特徴量
    - 学習データとテストデータを結合したデータセット
  - その他の特徴量
    - seen user model: Fold毎のtrain_data_set, val_data_set, test_data_setを結合したデータセット。val_data_setのスコアはnanにして特徴量の計算に使わない(学習の正解データとしては使う)。
    - unseen user model: Fold毎のseen_train_data_set, unseen_train_data_set, val_data_set, test_data_setを結合したデータセット。unseen_train_data_set, val_data_setのスコアはnanにして特徴量の計算に使わない(学習の正解データとしては使う)。
- 特徴量
  - 視聴情報に関する特徴量
    - user, animeをnodeとしたgraphに対してproneで計算したuserとanimeのベクトル(128次元)及び、userとanimeの類似度。
    - proneで計算したuserとanimeのベクトルに対してkmeansでクラスタリングしたクラスタ中心からの距離(クラスタ数100個)
    - user, genreをnodeとしたgraphに対してproneで計算したuserとgenresのベクトル(128次元)及び、userとgenresの類似度。  
    genre nodeにはgenres列を", "でsplitしたものを使う。含まれるgenreベクトルを平均してgenresベクトルにする。
    - 上記user, genresと同じ手順でuser, producerに対して計算したuserベクトルと類似度。producersベクトルは使わない。
    - 上記user, genresと同じ手順でuser, studioに対して計算したuserベクトルと類似度。studiosベクトルは使わない。
  - エンコーディング系特徴量 (特徴量の値がNanの場合は平均値で埋める)
    - userのtarget mean encoding, target std encoding, count encoding, 視聴したアニメ情報の数値特徴量の平均値エンコーディング。
    - 上記で計算したkmeansクラスタでのtarget mean encoding, target std encoding。
    - anime_id, アニメ情報カテゴリカルデータでのcount encoding, target mean encoding, target std encoding。
    - genres、producersそれぞれについてonehotに対してクラスタリングしたクラスタでのtarget mean encoding
    - 上2つのtarget mean encodingの値それぞれに対してuser scoreを引き、それをuser毎にmean encoding。  
    上2つの特徴量それぞれについてuser毎にmean encoding。
    - アニメ情報の数値そのまま
  - スコア考慮系Embedding特徴量 (ScoreがNaNの場合は平均値で埋めてからEmbeddingを計算する)
    - 上記のuser, anime graphで計算したベクトルに対して、userの隣接ノードベクトルをscoreの昇順pctランク**2を重みとして加重平均したベクトル。
    - 上記のuser, anime graphで計算したベクトルに対して、userの隣接ノードベクトルをscoreの降順pctランク**2を重みとして加重平均したベクトル。
    - scoreの昇順pctランク*10個だけ仮想的にanime nodeを増やしたuser, anime graphに対してproneで計算したuserベクトル(256次元)
    - scoreの降順pctランク*10個だけ仮想的にanime nodeを増やしたuser, anime graphに対してproneで計算したuserベクトル(256次元)

### 学習モデル
- 上記のseen user用CV (5 fold)、unseen user用CV (5 fold)のデータセットに対してxgboostで学習。

## 効かなかったこと
- unseen user modelの各Foldの学習データセットを、さらにGroupKFoldで分割して特徴量を計算しoof特徴量を学習データセットにするのを試したが悪化した。スコア考慮proneのembeddingベクトルに一貫性がなくなってうまく学習できかなった？

## やり残したこと
- user, animeのgraphに対してのみクラスタリング、スコア重み平均、スコア考慮proneを適用できなかったので、他のproducer等に対してもやりたかった。

## 感想
- proneは計算が速くてめちゃくちゃ良い。他のembedding手法は試していないので性能が良いかは未確認。
- scoreを考慮したproneをもっとスマートにしたい。edge weightやdirectionを考慮する方法があれば誰か教えてください。
- target encodingがかなり効いていたので、target encodingを使わずに高いスコアを出せるようで驚いた。
- NN好きなのでオレオレNNで勝負したかったが、お試しのGBDTが伸びたのでやれなかった。残念。
