# Extract the first GitHub-flavored Markdown pipe table from a file and emit CSV.
# Assumes the first table is a "|"-delimited table and ends when non-table text starts.

function trim(s) {
  sub(/^[ \t]+/, "", s)
  sub(/[ \t]+$/, "", s)
  return s
}

function esc(s) {
  gsub(/\r/, "", s)
  gsub(/<br[ \t]*\/?[ \t]*>/, "\\n", s)
  gsub(/"/, """" , s)
  return "\"" s "\""
}

BEGIN {
  in_table = 0
}

{
  if (!in_table) {
    if ($0 ~ /^\|/) {
      in_table = 1
    } else {
      next
    }
  }

  if ($0 !~ /^\|/) {
    exit
  }

  # Skip the Markdown header separator row like: | --- | ---: |
  if ($0 ~ /^\|[- :|]+\|[ \t]*$/ && $0 ~ /---/) {
    next
  }

  out = ""
  for (i = 2; i <= NF - 1; i++) {
    field = esc(trim($i))
    out = out (i == 2 ? "" : ",") field
  }
  print out
}
