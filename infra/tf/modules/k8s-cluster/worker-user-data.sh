#!/bin/bash
set -euxo pipefail

exec > >(tee /var/log/worker-user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting worker node initialization"

# --------------------------------------------------
# 1. Disable swap
# --------------------------------------------------

swapoff -a
sed -i '/ swap / s/^/#/' /etc/fstab

# --------------------------------------------------
# 2. Install basic dependencies
# --------------------------------------------------

apt-get update

DEBIAN_FRONTEND=noninteractive apt-get install -y \
  curl \
  apt-transport-https \
  ca-certificates \
  gpg \
  awscli

mkdir -p -m 755 /etc/apt/keyrings

# --------------------------------------------------
# 3. Load required kernel modules
# --------------------------------------------------

cat <<EOF > /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

modprobe overlay
modprobe br_netfilter

# --------------------------------------------------
# 4. Configure Kubernetes networking requirements
# --------------------------------------------------

cat <<EOF > /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sysctl --system

# --------------------------------------------------
# 5. Add Kubernetes repository
# Must match the Control Plane minor version
# --------------------------------------------------

KUBERNETES_VERSION="v1.34"

curl -fsSL \
  "https://pkgs.k8s.io/core:/stable:/${KUBERNETES_VERSION}/deb/Release.key" \
  | gpg --batch --yes --dearmor \
  -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo \
  "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/${KUBERNETES_VERSION}/deb/ /" \
  > /etc/apt/sources.list.d/kubernetes.list

# --------------------------------------------------
# 6. Add CRI-O repository
# Must match the Kubernetes minor version
# --------------------------------------------------

CRIO_VERSION="v1.34"

curl -fsSL \
  "https://download.opensuse.org/repositories/isv:/cri-o:/stable:/${CRIO_VERSION}/deb/Release.key" \
  | gpg --batch --yes --dearmor \
  -o /etc/apt/keyrings/cri-o-apt-keyring.gpg

echo \
  "deb [signed-by=/etc/apt/keyrings/cri-o-apt-keyring.gpg] https://download.opensuse.org/repositories/isv:/cri-o:/stable:/${CRIO_VERSION}/deb/ /" \
  > /etc/apt/sources.list.d/cri-o.list

# --------------------------------------------------
# 7. Install CRI-O and Kubernetes components
# --------------------------------------------------

apt-get update

DEBIAN_FRONTEND=noninteractive apt-get install -y \
  cri-o \
  kubelet \
  kubeadm

apt-mark hold kubelet kubeadm

# --------------------------------------------------
# 8. Start CRI-O and enable kubelet
# --------------------------------------------------

systemctl daemon-reload
systemctl enable --now crio
systemctl is-active --quiet crio

# kubelet waits until kubeadm join creates its configuration
systemctl enable kubelet

# --------------------------------------------------
# 9. Prevent duplicate join
# --------------------------------------------------

if [ -f /etc/kubernetes/kubelet.conf ]; then
  echo "Worker is already joined to the Kubernetes cluster"
  exit 0
fi

# --------------------------------------------------
# 10. Retrieve kubeadm join command from SSM
# --------------------------------------------------

AWS_REGION="us-east-1"
JOIN_PARAMETER="/k8s/adan/join-command"
JOIN_COMMAND=""

# Retry because:
# 1. The Control Plane User Data may still be running.
# 2. The SSM parameter may still contain PLACEHOLDER.
# 3. IAM credentials may not yet be available.
for attempt in $(seq 1 60); do
  echo "Retrieving kubeadm join command: attempt ${attempt}/60"

  JOIN_COMMAND=$(aws ssm get-parameter \
    --name "${JOIN_PARAMETER}" \
    --with-decryption \
    --region "${AWS_REGION}" \
    --query "Parameter.Value" \
    --output text 2>/dev/null || true)

  if [ -n "${JOIN_COMMAND}" ] \
    && [ "${JOIN_COMMAND}" != "None" ] \
    && [ "${JOIN_COMMAND}" != "PLACEHOLDER" ]; then
    echo "Valid kubeadm join command retrieved"
    break
  fi

  JOIN_COMMAND=""
  sleep 15
done

if [ -z "${JOIN_COMMAND}" ]; then
  echo "ERROR: Could not retrieve a valid kubeadm join command from SSM"
  exit 1
fi

# --------------------------------------------------
# 11. Join the Kubernetes cluster
# --------------------------------------------------

echo "Joining worker to the Kubernetes cluster"

${JOIN_COMMAND} \
  --cri-socket=unix:///var/run/crio/crio.sock

systemctl restart kubelet
systemctl is-active --quiet kubelet

echo "Worker joined the Kubernetes cluster successfully"