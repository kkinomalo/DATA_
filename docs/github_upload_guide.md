# GitHub 업로드 방법

## 방법 1. VS Code 터미널에서 올리기

프로젝트 폴더에서 아래 명령어를 순서대로 실행합니다.

```powershell
git init
git add .
git commit -m "Add heart attack risk 1D CNN project"
git branch -M main
git remote add origin https://github.com/본인아이디/heart-attack-risk-1d-cnn.git
git push -u origin main
```

`본인아이디` 부분은 본인의 GitHub 아이디로 바꿔야 합니다.

## 방법 2. GitHub 웹사이트에서 직접 올리기

1. GitHub 접속
2. New repository 클릭
3. Repository name 입력: `heart-attack-risk-1d-cnn`
4. Public 또는 Private 선택
5. Create repository 클릭
6. Add file → Upload files 클릭
7. 이 프로젝트 폴더 안의 파일들을 드래그해서 업로드
8. Commit changes 클릭

## GitHub에 올릴 파일

올려야 하는 파일:

```text
main.py
DATA_fixed.py
requirements.txt
README.md
REPORT.md
.gitignore
docs/github_upload_guide.md
outputs/.gitkeep
```

올리지 않는 파일:

```text
.venv/
__pycache__/
*.keras
outputs/*.png
outputs/*.txt
```

모델 파일과 결과 이미지는 용량이 커질 수 있으므로 기본적으로 제외합니다. 보고서에 결과 이미지를 넣어야 한다면 `outputs/confusion_matrix.png`, `outputs/loss_graph.png`, `outputs/accuracy_graph.png`만 따로 올려도 됩니다.
