from llama_index.tools import FunctionTool
import os

note_file = os.path.join("data", "note.txt") # This is the file where the note will be saved


def save_note(note: str):
    if not os.path.exists(note_file):
        open(note_file, 'w')

    with open(note_file, 'w') as f:
        f.writelines([note + "\n"]) # Write the note to the file

    return "note saved successfully!"

note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name='note_saver',
    description='Save a note to a file',
)