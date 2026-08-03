#!/bin/bash

set -euxo pipefail

exec > >(tee /var/log/control-plane-user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

KUBERNETES_VERSION="v1.34"
CRIO_VERSION="v1.34"

echo "Starting control plane initialization"

# --------------------------------------------------
# 1. Disable swap
# --------------------------------------------------

swapoff -a
sed -i '/ swap / s/^/#/' /etc/fstab

# --------------------------------------------------
# 2. Load required kernel modules
# --------------------------------------------------

cat <<EOF > /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

modprobe overlay
modprobe br_netfilter

# --------------------------------------------------
# 3. Configure Kubernetes networking requirements
# --------------------------------------------------

cat <<EOF > /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sysctl --system

# --------------------------------------------------
# 4. Install repository dependencies
# --------------------------------------------------

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates \
  curl \
  gpg \
  software-properties-common \
  awscli

mkdir -p -m 755 /etc/apt/keyrings

# --------------------------------------------------
# 5. Add Kubernetes repository
# --------------------------------------------------

curl -fsSL \
  "https://pkgs.k8s.io/core:/stable:/${KUBERNETES_VERSION}/deb/Release.key" \
  | gpg --batch --yes --dearmor \
  -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo \
  "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/${KUBERNETES_VERSION}/deb/ /" \
  > /etc/apt/sources.list.d/kubernetes.list

# --------------------------------------------------
# 6. Add CRI-O repository
# --------------------------------------------------

curl -fsSL \
  "https://download.opensuse.org/repositories/isv:/cri-o:/stable:/${CRIO_VERSION}/deb/Release.key" \
  | gpg --batch --yes --dearmor \
  -o /etc/apt/keyrings/cri-o-apt-keyring.gpg

echo \
  "deb [signed-by=/etc/apt/keyrings/cri-o-apt-keyring.gpg] https://download.opensuse.org/repositories/isv:/cri-o:/stable:/${CRIO_VERSION}/deb/ /" \
  > /etc/apt/sources.list.d/cri-o.list

# --------------------------------------------------
# 7. Install CRI-O and Kubernetes
# --------------------------------------------------

apt-get update

DEBIAN_FRONTEND=noninteractive apt-get install -y \
  cri-o \
  kubelet \
  kubeadm \
  kubectl

apt-mark hold kubelet kubeadm kubectl

# --------------------------------------------------
# 8. Start services
# --------------------------------------------------

systemctl daemon-reload
systemctl enable --now crio
systemctl enable --now kubelet

systemctl is-active --quiet crio

# --------------------------------------------------
# 9. Detect the EC2 private IP
# --------------------------------------------------

PRIVATE_IP=$(hostname -I | awk '{print $1}')

echo "Control plane private IP: ${PRIVATE_IP}"

# --------------------------------------------------
# 10. Initialize Kubernetes control plane
# --------------------------------------------------

if [ ! -f /etc/kubernetes/admin.conf ]; then
  kubeadm init \
    --kubernetes-version="v1.34.10" \
    --apiserver-advertise-address="${PRIVATE_IP}" \
    --cri-socket=unix:///var/run/crio/crio.sock
else
  echo "Control plane is already initialized"
fi

# --------------------------------------------------
# 11. Configure kubectl for ubuntu user
# --------------------------------------------------

mkdir -p /home/ubuntu/.kube
install -o ubuntu -g ubuntu -m 600 \
  /etc/kubernetes/admin.conf \
  /home/ubuntu/.kube/config

export KUBECONFIG=/etc/kubernetes/admin.conf

until kubectl get --raw='/readyz' >/dev/null 2>&1; do
  echo "Waiting for Kubernetes API server..."
  sleep 5
done

echo "Control plane initialization completed successfully"

# --------------------------------------------------
# 12. Generate and store worker join command
# --------------------------------------------------

JOIN_COMMAND=$(kubeadm token create --ttl 0 --print-join-command)

aws ssm put-parameter \
  --name "/k8s/adan/join-command" \
  --type "SecureString" \
  --value "${JOIN_COMMAND}" \
  --overwrite \
  --region "us-east-1"

echo "Worker join command stored in SSM Parameter Store"