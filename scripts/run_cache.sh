#!/bin/bash

# Default to cache_commands.txt if no argument provided
COMMANDS_FILE=${1:-cache_commands.txt}

# Check if commands file exists
if [ ! -f "$COMMANDS_FILE" ]; then
    echo "Error: Commands file '$COMMANDS_FILE' not found!"
    exit 1
fi

echo "++++++++++++++++++++++++++++++++++++"
echo "+++ start cache generation script +++"
echo "+++ Reading from: $COMMANDS_FILE"
echo "++++++++++++++++++++++++++++++++++++"
echo ""

# Create unique log file based on commands file name
LOG_FILE="cache_error_log_$(basename "$COMMANDS_FILE" .txt).txt"
rm -f "$LOG_FILE"

# Read commands from specified file and execute them
echo "+++ executing commands in series"
echo ""

while IFS= read -r command || [ -n "$command" ]; do
    # Skip empty lines
    if [ -z "$command" ]; then
        continue
    fi

    echo "executing command: $command"
    echo ""

    # Run the Python command and display the output in real time,
    # also capture the output
    output=$(mktemp)
    bash -c "$command" | tee "$output"

    # Check if command was successful
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        # Command failed. Log to error file
        echo "FAILED command: $command" >> "$LOG_FILE"
        cat "$output" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
    fi
    rm "$output"

done < "$COMMANDS_FILE"

echo "++++++++++++++++++++++++++++++++++"
echo "+++ end cache generation script +++"
echo "++++++++++++++++++++++++++++++++++"

# Report any errors
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "⚠️  Some commands failed. Check $LOG_FILE for details."
else
    echo ""
    echo "✓ All commands completed successfully!"
fi