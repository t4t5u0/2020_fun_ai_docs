import pathlib

path = pathlib.Path(__file__).resolve()
print(path)
#　1つ上に戻りたいとき
print(path.parent)

path = pathlib.Path.cwd()
print(path)
