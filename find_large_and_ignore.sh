#!/bin/bash
echo "Searching for files > 50MB..."
find . -type f -size +50M > large_files.txt

echo "Appending to .gitignore..."
cat large_files.txt >> .gitignore
sort -u .gitignore -o .gitignore

echo "Removing from Git index..."
while read file; do
  git rm --cached "$file" 2>/dev/null
done < large_files.txt

git commit -m "Remove large files and update .gitignore"
rm large_files.txt

