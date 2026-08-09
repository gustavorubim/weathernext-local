#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
install_root="/opt/weathernext-local"
uv_bin="${install_root}/bin/uv"
uv_version="0.12.3"
uv_sha256="600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101"
uv_binary_sha256="729d27dbea534ee540a2d3ef43a62fa1a10af7fcbb6d57a70d5859509f624578"
uvx_binary_sha256="291766f4eed0c497b5585e0a0cbadfd8fc14070bfd17cbead96d1b8b63cf3ac4"
uv_asset="uv-x86_64-unknown-linux-gnu.tar.gz"
runtime_user="weathernext"

for managed_path in \
  "${install_root}" \
  "${install_root}/bin" \
  "${install_root}/bin/uv" \
  "${install_root}/bin/uvx" \
  "${install_root}/cache" \
  "${install_root}/python" \
  "${install_root}/python-bin" \
  "${install_root}/uv-cache" \
  "${install_root}/venv"; do
  if [[ -L "${managed_path}" ]]; then
    printf 'Refusing privileged setup through symbolic link: %s\n' "${managed_path}" >&2
    exit 1
  fi
done

mkdir -p \
  "${install_root}/bin" \
  "${install_root}/python" \
  "${install_root}/python-bin" \
  "${install_root}/uv-cache"
if [[ "$(realpath -e "${install_root}")" != "${install_root}" \
   || "$(realpath -e "${install_root}/bin")" != "${install_root}/bin" ]]; then
  printf 'Refusing unexpected resolved installation path under /opt.\n' >&2
  exit 1
fi

installed_uv_sha256=""
installed_uvx_sha256=""
if [[ -x "${uv_bin}" ]]; then
  installed_uv_sha256="$(sha256sum "${uv_bin}" 2>/dev/null | awk '{print $1}' || true)"
fi
if [[ -x "${install_root}/bin/uvx" ]]; then
  installed_uvx_sha256="$(sha256sum "${install_root}/bin/uvx" 2>/dev/null | awk '{print $1}' || true)"
fi

if [[ "${installed_uv_sha256}" != "${uv_binary_sha256}" \
   || "${installed_uvx_sha256}" != "${uvx_binary_sha256}" ]]; then
  installer_dir="$(mktemp -d)"
  trap 'rm -rf "${installer_dir}"' EXIT
  archive="${installer_dir}/${uv_asset}"
  curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location \
    "https://github.com/astral-sh/uv/releases/download/${uv_version}/${uv_asset}" \
    --output "${archive}"
  printf '%s  %s\n' "${uv_sha256}" "${archive}" | sha256sum --check --status
  tar -xzf "${archive}" -C "${installer_dir}"
  install -m 0755 \
    "${installer_dir}/uv-x86_64-unknown-linux-gnu/uv" \
    "${install_root}/bin/uv"
  install -m 0755 \
    "${installer_dir}/uv-x86_64-unknown-linux-gnu/uvx" \
    "${install_root}/bin/uvx"
fi
chown root:root "${install_root}" "${install_root}/bin" "${uv_bin}" "${install_root}/bin/uvx"
chmod 0755 "${install_root}" "${install_root}/bin" "${uv_bin}" "${install_root}/bin/uvx"

if ! id -u "${runtime_user}" >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash --user-group "${runtime_user}"
fi
runtime_uid="$(id -u "${runtime_user}")"
runtime_gid="$(id -g "${runtime_user}")"
runtime_group="$(id -gn "${runtime_user}")"
runtime_groups="$(id -Gn "${runtime_user}")"
runtime_home="$(getent passwd "${runtime_user}" | cut -d: -f6)"
runtime_shell="$(getent passwd "${runtime_user}" | cut -d: -f7)"
if [[ "${runtime_uid}" == "0" \
   || "${runtime_gid}" == "0" \
   || "${runtime_group}" != "${runtime_user}" \
   || "${runtime_groups}" != "${runtime_user}" \
   || "${runtime_home}" != "/home/${runtime_user}" \
   || "${runtime_shell}" != "/bin/bash" \
   || ! -d "${runtime_home}" \
   || -L "${runtime_home}" ]]; then
  printf 'Refusing unsafe pre-existing runtime identity for %s.\n' "${runtime_user}" >&2
  exit 1
fi
runtime_home_uid="$(stat -c '%u' "${runtime_home}")"
runtime_home_gid="$(stat -c '%g' "${runtime_home}")"
if [[ "${runtime_home_uid}" != "${runtime_uid}" \
   || "${runtime_home_gid}" != "${runtime_gid}" ]]; then
  printf 'Refusing unsafe pre-existing runtime identity for %s.\n' "${runtime_user}" >&2
  exit 1
fi
install -d -m 0755 -o "${runtime_user}" -g "${runtime_user}" "${install_root}/cache"
chown -R "${runtime_user}:${runtime_user}" \
  "${install_root}/cache" \
  "${install_root}/python" \
  "${install_root}/python-bin" \
  "${install_root}/uv-cache"
if [[ -e "${install_root}/venv" ]]; then
  chown -R "${runtime_user}:${runtime_user}" "${install_root}/venv"
fi

uv_cache_dir="${install_root}/uv-cache"
uv_python_bin_dir="${install_root}/python-bin"
uv_python_install_dir="${install_root}/python"
uv_project_environment="${install_root}/venv"

run_as_runtime() {
  runuser -u "${runtime_user}" -- env \
    HOME="/home/${runtime_user}" \
    XDG_CACHE_HOME="${install_root}/cache" \
    UV_CACHE_DIR="${uv_cache_dir}" \
    UV_PYTHON_BIN_DIR="${uv_python_bin_dir}" \
    UV_PYTHON_INSTALL_DIR="${uv_python_install_dir}" \
    UV_PROJECT_ENVIRONMENT="${uv_project_environment}" \
    "$@"
}

run_as_runtime "${uv_bin}" python install 3.12.13
if [[ -x "${uv_project_environment}/bin/python" ]]; then
  environment_python="$(readlink -f "${uv_project_environment}/bin/python")"
  if [[ "${environment_python}" != "${uv_python_install_dir}/"* ]]; then
    resolved_environment="$(realpath -m "${uv_project_environment}")"
    if [[ "${resolved_environment}" != "/opt/weathernext-local/venv" ]]; then
      printf 'Refusing to replace unexpected environment path: %s\n' "${resolved_environment}" >&2
      exit 1
    fi
    rm -rf -- "${resolved_environment}"
  fi
fi
install -d -m 0755 -o "${runtime_user}" -g "${runtime_user}" "${uv_project_environment}"
run_as_runtime "${uv_bin}" sync \
  --project "${repo_root}" \
  --python 3.12.13 \
  --locked \
  --no-editable \
  --reinstall-package weathernext-local \
  --extra dev \
  --extra model

printf '\nWeatherNext Local environment is ready.\n'
printf 'Environment: %s\n' "${uv_project_environment}"
printf 'Runtime user: %s\n' "${runtime_user}"
run_as_runtime "${uv_project_environment}/bin/wnx" doctor
