"""Pacote shapes: reexporta utilitários de UI.

Preferimos reexportar do módulo local `.design` (dentro do pacote) para manter
o isolamento e evitar conflitos com um `design.py` na raiz do projeto.
"""

try:
	from .design import apply_custom_styles, create_header  # type: ignore
except Exception:
	# Mantém o pacote importável mesmo se houver erro de import momentâneo.
	apply_custom_styles = None  # type: ignore
	create_header = None  # type: ignore

__all__ = [
	"apply_custom_styles",
	"create_header",
]
