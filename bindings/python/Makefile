.PHONY: deps_table_update modified_only_fixup extra_style_checks quality style fixup fix-copies test test-examples

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := tests py_src

modified_only_fixup:
	$(eval modified_py_files := $(shell python utils/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		black --preview $(modified_py_files); \
		isort $(modified_py_files); \
		flake8 $(modified_py_files); \
	else \
		echo "No library .py files were modified"; \
	fi


quality:
	black --check --preview $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
	doc-builder style src/transformers docs/source --max_len 119 --check_only --path_to_docs docs/source

style:
	black --preview $(check_dirs)
	isort $(check_dirs)

# Super fast fix and check target that only works on relevant modified files since the branch was made

fixup: modified_only_fixup

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/
