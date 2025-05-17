# ✅ Project 10k → 1M 開発者向けセットアップ・再開チェックリスト（VS Code + Python）

## 📁 プロジェクトフォルダ

```
C:\Projects\project_10k_to_1m
```

---

## ✅ 作業再開時のルーティン（毎回やること）

### ① VS Code を起動し、プロジェクトを開く

* 「ファイル → フォルダを開く」
* パス：`C:\Projects\project_10k_to_1m`

### ② VS Code でターミナルを開く

* 「ターミナル → 新しいターミナル」
* 初期状態で PowerShell が開くはず

### ③ PowerShell の実行ポリシーを一時的に許可（必要なときのみ）

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

* メッセージが出たら `A` を入力して Enter

### ④ 仮想環境を有効化

```powershell
. .\venv\Scripts\activate
```

* 成功するとプロンプトが `(venv)` になる

### ⑤ ライブラリをインストール（初回 or ライブラリ追加時）

```powershell
pip install -r requirements.txt
```

### ⑥ スクリプトを実行

```powershell
python src/data/Get_OHLCV.py
```

または、該当の `.py` ファイルを VS Code 上で開いて「▶ 実行」

---

## ✅ ta-lib のインストール（Windows + Python 3.10.11）

### 手順：

1. Gohlke’s site または SourceForge から以下の `.whl` を入手：

   * `TA_Lib-0.4.28-cp310-cp310-win_amd64.whl`
2. ダウンロードフォルダに移動：

```powershell
cd $HOME\Downloads
```

3. 仮想環境をアクティブにした上でインストール：

```powershell
pip install TA_Lib-0.4.28-cp310-cp310-win_amd64.whl
```

---

## ✅ トラブルシューティングメモ

| 問題                  | 解決策                                    |
| ------------------- | -------------------------------------- |
| activate エラー        | `Set-ExecutionPolicy` を実行              |
| ta-lib エラー          | `.whl` ファイルを個別にダウンロードしてインストール          |
| ライブラリ不足             | `pip install -r requirements.txt` を再実行 |
| Colab 側でファイルが見つからない | `/content/drive/MyDrive/` 配下に正しく配置する   |

---

## 🧠 ワンポイントメモ

* VS Code の右下の Python バージョン表示で仮想環境が選ばれているか確認
* requirements.txt を編集したら必ず再インストール
* データ取得用 MT5 は起動した状態で実行する

---

以上、毎回の作業を手戻りなく進めるためのチェックリストです。
開発が進んだら、必要に応じてこのファイルに項目を追加してください。
