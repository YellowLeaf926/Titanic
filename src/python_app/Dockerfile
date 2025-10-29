##Load from python
FROM python
##copy all files
COPY python_app/ .
COPY data/ .

##Run requirements to install packages
RUN pip install --no-cache-dir -r requirements.txt
#When you run, run app.py
CMD ["python3", "app.py"]