FROM mambaorg/micromamba:1.5.6 as builder

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN --mount=type=cache,target=/opt/conda/pkgs micromamba install -y -n base -f /tmp/env.yaml

# add some packages 
USER root
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

#USER $MAMBA_USER
# Download and install nist_mni_pipelines
RUN curl -q -L https://github.com/NIST-MNI/nist_mni_pipelines/archive/refs/heads/ray_skull.tar.gz --output /tmp/nist_mni_pipelines.tar.gz && \
    mkdir -p /opt/pipeline && \
    tar -xzvf /tmp/nist_mni_pipelines.tar.gz -C /opt/pipeline --strip-components=1  

# TODO: Download models
RUN --mount=type=cache,target=/opt/archive \
    curl -q -L https://packages.bic.mni.mcgill.ca/mni-models/icbm152/mni_icbm152_nlin_sym_09c_minc2.zip \
          --output /opt/archive/mni_icbm152_nlin_sym_09c_minc2.zip && \
    mkdir -p /opt/models/icbm152_model_09c && \
    unzip /opt/archive/mni_icbm152_nlin_sym_09c_minc2.zip -d /opt/models/icbm152_model_09c

RUN chown $MAMBA_USER:$MAMBA_USER -R /opt/models /opt/pipeline 

#install local models
#USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER models/ipl_bison_1.3.0 /opt/models/ipl_bison_1.3.0
COPY --chown=$MAMBA_USER:$MAMBA_USER models/wmh_bison_1.3.0 /opt/models/wmh_bison_1.3.0
COPY --chown=$MAMBA_USER:$MAMBA_USER models/synthstrip /opt/models/synthstrip

# make a clean image without temporary files
FROM mambaorg/micromamba:1.5.6 
COPY --from=builder /opt/conda    /opt/conda
COPY --from=builder /opt/pipeline /opt/pipeline
COPY --from=builder /opt/models   /opt/models

# Distroless for execution
# FROM gcr.io/distroless/base-debian12

# ARG MAMBA_USER=mambauser
# ARG MAMBA_USER_ID=57439
# ARG MAMBA_USER_GID=57439

# COPY --from=micromamba /opt/conda /opt/conda
# COPY --from=micromamba /bin/micromamba /bin/micromamba

# COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
# COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
# COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
# COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
# COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

# RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
#     /usr/local/bin/_dockerfile_setup_root_prefix.sh

# USER $MAMBA_USER

# SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
ENV PYTHONPATH=/opt/pipeline
