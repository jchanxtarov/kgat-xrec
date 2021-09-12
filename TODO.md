# TODO
## 徐々に進める
- [x] （比較）BPRMF の実装
- [x] 解釈のためのパス探索のためのスクリプトを追加
- ~~[ ] plsa による圧縮と復元機能の追加~~
- [x] user ごとの推薦アイテムの記録の仕組みを追加
- [x] for use_user_attribute
- [x] torch seed の位置を確認
- [x] .cpu(), transpose, squeeze 周りの整理
- [x] gpu(device) 周りの修正
- [x] 型チェックを全体に反映
- [ ] apply formatter

## 余裕があれば
- [x] main.py に predict を入れる
- [ ] （比較）NFM の実装
- [ ] （比較）CFKG の実装
- [ ] metrics を numpy -> torch base に

## package 化するなら
- [ ] logging の初期設定を Dataset の中かどこかですべき
- [ ] loader を model 中かどこかで初期化（loader, model の統合）

## 常時
- [ ] 精度の向上のためのパラメータ探索
