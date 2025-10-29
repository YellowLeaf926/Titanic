FROM rocker/r-base:4.3.1

# Create app directory
RUN mkdir /R_app

# Set working directory
WORKDIR /R_app

# Copy your app files into the image
COPY R_app/ .
COPY data/ .

# Install R packages
RUN Rscript install_packages.R

# Run the app
CMD ["Rscript", "app.R"]