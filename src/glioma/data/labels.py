"""Frozen IDH/MGMT label mapping and cross-assertions against the diagnosis column.

Never use `Final pathologic diagnosis`, `WHO CNS Grade` or `1p/19q` as model inputs -
see CLAUDE.md §2 rule 4. The diagnosis string contains the IDH label verbatim.
"""
