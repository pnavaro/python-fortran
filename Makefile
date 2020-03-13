all:
	jupytext --to markdown --execute *.py
	mv *.md docs/
