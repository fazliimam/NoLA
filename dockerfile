# Use the official miniconda3 base image
FROM continuumio/miniconda3

# Copy the environment.yml file into the Docker image
COPY alp_rs.yml .

# Create the Conda environment inside the Docker image
RUN conda env create -f alp_rs.yml

# Make sure the environment is activated by default
# This ensures that the Conda environment is activated for any subsequent RUN commands
RUN echo "conda activate alp_rs" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Set the default command to run when the container starts
CMD ["bash"]
