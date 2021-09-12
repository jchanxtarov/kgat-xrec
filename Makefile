default: | help

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "   run              to run using sample dataset with saving all files"
	@echo "   dry-run          to run using sample dataset without saving any files"
	@echo "   check            to type check"
	@echo "   setup            to setup to run"
	@echo "   search-path      to search path from {userid} to {itemid}"
	@echo "   dry-run-pretrain to run using pretrained params without saving any files"

# run using sample dataset with saving any logfiles
run:
	poetry run python src/main.py --use_user_attribute --save_log --save_model --predict --save_recommended_items

# run using sample dataset without saving any files
dry-run:
	poetry run python src/main.py --use_user_attribute --predict

# type check
check:
	poetry run mypy src/main.py

# setup to run
setup:
	poetry install

# ex: $ make search-path userid=0 itemid=29727
search-path:
	poetry run python src/scripts/search_path.py --userid $(userid) --itemid $(itemid)

# run using pretrained params without saving any files
dry-run-pretrain:
	poetry run python src/main.py --use_user_attribute --predict --use_pretrain
