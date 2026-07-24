.PHONY: install run test lint docker-build docker-run compose-up

install:
	pip install -r requirements-dev.txt

run:
	streamlit run app/streamlit_app.py

test:
	pytest --cov=app --cov-report=term-missing tests/

lint:
	flake8 app tests scripts

docker-build:
	docker build -t delivery-delay-app .

docker-run:
	docker run -p 8501:8501 -v $(PWD)/model:/app/model delivery-delay-app

compose-up:
	docker compose up --build
