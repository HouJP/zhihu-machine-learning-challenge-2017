#! /bin/bash

function init_dir() {
    project_pt=${1}

    out_pt=${project_pt}/out/
    log_pt=${project_pt}/log/
}

init_dir ${1}