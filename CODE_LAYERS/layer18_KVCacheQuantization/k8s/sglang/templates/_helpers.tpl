{{/*
Expand the chart name.
*/}}
{{- define "sglang.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Full name: release-chart (capped at 63 chars).
*/}}
{{- define "sglang.fullname" -}}
{{- $name := .Chart.Name }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Chart label (name + version).
*/}}
{{- define "sglang.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels applied to every resource.
*/}}
{{- define "sglang.labels" -}}
helm.sh/chart: {{ include "sglang.chart" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}

{{/*
Namespace to deploy into.
*/}}
{{- define "sglang.namespace" -}}
{{ .Values.namespace.name }}
{{- end }}

{{/*
Name of the HuggingFace token secret.
If existingSecret is set, use that; otherwise use the chart-managed secret.
*/}}
{{- define "sglang.hfSecretName" -}}
{{- if .Values.model.hfToken.existingSecret }}
{{- .Values.model.hfToken.existingSecret }}
{{- else }}
{{- include "sglang.fullname" . }}-hf-token
{{- end }}
{{- end }}

{{/*
Name of the PVC for model weights.
*/}}
{{- define "sglang.pvcName" -}}
{{- include "sglang.fullname" . }}-model-cache
{{- end }}

{{/*
Name of the router ServiceAccount (used for K8s API access).
*/}}
{{- define "sglang.routerServiceAccount" -}}
{{ .Values.router.serviceAccountName }}
{{- end }}
