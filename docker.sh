#!/bin/bash
CONTAINER_TAG="stylegan3"
CONTAINER_PROJECT_NAME="ian" # Suggest
CONTAINER_NAMETAG="$CONTAINER_PROJECT_NAME-$CONTAINER_TAG" # Suggest

ARG_USER="-u $(whoami)"
COMMAND='/bin/bash'
POSITIONAL_ARGS=()
DOCKER_BUILD=false
TENSORBOARD=false

# Get command line arguments
OPTIONS=bfru:e:htT   # -b build, -r run, -e execute, -h help
OPTIONS_LONG=build,run,exec,help
OPTIND=1 # Holds the number of options parsed by the last call to getopts. Reset in case getopts has been used previously in the shell
while getopts $OPTIONS opt; do
    case "${opt}" in
        b) # Build
            DOCKER_BUILD=true
            ;;
        r) # Run as Root
            ARG_USER=' -u root '
            ;;
        u) # Run as user
            ARG_USER=" -u $OPTARG "
            ;;
        c) # Execute command
            COMMAND=$OPTARG
            ;;
        f) # No docker cache
            NO_CACHE=' --no-cache '
            ;;
        t|T) # Tensorboard
            TENSORBOARD=true
            ;;
        \?) # Invalid option
            echo "Invalid option: -$opt" >&2
            exit 1
            ;;
        h) # Help
            echo "Need help? There is no help. Read the code."
            exit 0
            ;;
    esac
done
shift $((OPTIND-1)) # remove options that have already been handled from $@


if [[ $TENSORBOARD == true ]]; then
    echo "<?> STARTING TENSORBOARD <?>"
    docker compose --project-name $CONTAINER_PROJECT_NAME exec $CONTAINER_TAG pkill -f tensorboard
    tmux new-session -s "Tensorboard" "docker compose --project-name $CONTAINER_PROJECT_NAME exec $ARG_USER $CONTAINER_TAG tensorboard --logdir /workspace/logs --bind_all --port 6006 --samples_per_plugin images=999999 --max_reload_threads 4"
    exit 0
fi

if [[ $DOCKER_BUILD == true ]]; then
    echo "<?> STARTING DOCKER BUILD <?>"
    docker nvidia/cuda:12.3.2-runtime-ubuntu22.04
    docker compose --project-name $CONTAINER_PROJECT_NAME --file .docker/compose.yml stop
    docker compose --project-name $CONTAINER_PROJECT_NAME --file .docker/compose.yml rm -f
    docker compose --file .docker/compose.yml pull
    docker compose --project-name $CONTAINER_PROJECT_NAME -f .docker/compose.yml build $NO_CACHE --build-arg USER_NAME=$(whoami) --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)

    if [[ $? -ne 0 ]]; then
        echo "<?> DOCKER BUILD FAILED <?>"
        exit 1
    fi

    docker compose --project-name $CONTAINER_PROJECT_NAME --file .docker/compose.yml up --detach --no-recreate

    # Install custom libraries
    docker compose --project-name $CONTAINER_PROJECT_NAME exec $ARG_USER -w /workspace/libs/Parametric_SkyModels $CONTAINER_TAG python -m pip install .
    # docker compose --project-name $CONTAINER_PROJECT_NAME exec $ARG_USER -w /workspace/libs/research-utils $CONTAINER_TAG python -m pip install .
fi

echo "<?> STARTING SHELL IN CONTAINER <?>"
docker compose --project-name $CONTAINER_PROJECT_NAME exec $ARG_USER $CONTAINER_TAG $COMMAND