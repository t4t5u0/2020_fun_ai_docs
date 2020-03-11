---
marp: true
_backgroundColor: #333333 
paginate: true
_paginate: false
_color: white
---

# FUN AI 第1回 環境構築

## GitHub 篇

---

## 挙手

GitHub というサービスを知っている人

---

## 挙手

GitHub のアカウントを持っている人

---

# GitHubとは

公式より引用
>GitHubはソフトウェア開発のプラットフォームです。GitHubには8000万件以上ものプロジェクトがホスティングされており、2700万人以上のユーザーがプロジェクトを探したり、フォークしたり、コントリビュートしたりしています。

---

## なんのことかわからん🤔

---

GitHubには嬉しい機能がいくつかあり

- バージョン管理がしやすい(変更前に戻れる)

- 複数人で同じプロジェクトをすすめることができる

- GitHub上のプロジェクトを手元に持ってくるのが簡単

などなど

---

ということで、

## アカウントを作っていく

---

まずは公式サイトにアクセス
## [https://github.com](https://github.com)

---

アカウントを持っていない人は sign up をする
![h300](./fig/01/01_01.png)

---

### メールアドレスはあとから追加できるから、とりあえず学内メールで作るといいかもしれない。

### もちろん好きなもので作ってもらって構わない。

### ついでに sign in までしてしまおう。

---

次に、

## [このサイト](https://qiita.com/toshi-click/items/dcf3dd48fdc74c91b409)

にアクセスし、手順通りに進める

---

Windowsキー(スタートキー)を押し、アプリを検索できる状態にする。

powershell と入力し、Windows powershell を立ち上げる。

---

`git --version` と入力する

```Console
$ git --version
git version 2.23.0.windows.1
```

などと表示されればよい

---

同じく powershell 上で、

```Console
$ cd ~/Documents
$ git clone https://github.com/t4t5u0/2020_fun_ai_docs.git
```
と叩く

