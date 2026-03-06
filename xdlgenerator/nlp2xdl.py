import argparse
import json
import os
import sys

import numpy as np
import openai
from tqdm import tqdm

from constraint_bootstrap import ConstraintBootstrapTrainer, load_training_samples

wd = os.getcwd()
root_dir = "/".join(wd.split("/"))
sys.path.append(root_dir)
from verifier import verify

openai.api_key = os.environ["OPENAI_API_KEY"]


def prompt(instructions, description, max_tokens, task="\nConvert to XDL:\n", constraints=""):
    """prompt.

    Parameters
    ----------
    instructions :
        instructions
    description :
        description
    max_tokens :
        max_tokens
    task :
        task
    """
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=description +constraints+ "\nConvert to XDL:\n" + instructions,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"]


def generate_xdl(file_path, available_hardware=None, available_reagents=None):
    """generate_xdl.

    Parameters
    ----------
    file_path :
        file_path
    """
    instructions = open(file_path, "r").read()
    XDL = open("XDL_description.txt", "r").read()
    prev_instr = instructions
    correct_syntax = False
    errors = {}
    task = "\nConvert to XDL:\n"

    constraints=""
    if available_hardware!= None:
        hardware_str = ", ".join(available_hardware)[:-2]
        constraints = f"\nThe available Hardware is: {hardware_str}\n"
    if available_reagents!= None:
        reagents_str = ", ".join(available_reagents)[:-2]
        constraints += f"\nThe available Reagents are: {reagents_str}\n"
    for step in range(10):
        print(constraints+"\nConvert to XDL:\n" + instructions)
        try:
            gpt3_output = prompt(instructions, XDL, 1000, task, constraints)
        except openai.error.InvalidRequestError:
             gpt3_output = prompt(instructions, XDL, 750, task, constraints)
        print("gpt3 output:::")
        print(gpt3_output)
        print("******")
        gpt3_output = gpt3_output[gpt3_output.index("<XDL>"):]
        compile_correct = verify.verify_xdl(gpt3_output, available_hardware, available_reagents)
        errors[step] = {
            "errors": compile_correct,
            "instructions": instructions,
            "gpt3_output": gpt3_output,
        }
        if len(compile_correct) == 0:
            correct_syntax = True
            break
        else:
            error_list = set()
            for ii in compile_correct:
                for jj in ii["errors"]:
                    error_list.add(jj)
            abl=2
            if abl== 0:
                error_message = "\n{}\nThis XDL was not correct. Please fix the errors.".format(
                    gpt3_output
                )
            elif abl==1:
                error_message = "These are XDL errors.\n{}\nPlease fix the errors.".format(
                    "\n".join(list(error_list))
                )
            #### BEST ONE:
            elif abl==2:
                error_message = "\n{}\nThis XDL was not correct. These were the errors\n{}\nPlease fix the errors.".format(
                    gpt3_output,
                    "\n".join(list(error_list))
                )
            instructions = prev_instr + " " + error_message

    if correct_syntax:
        return correct_syntax, gpt3_output, errors
    else:
        return correct_syntax, "The correct XDL could not be generated.", errors


def suggest_constraints_from_failure(payload):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are improving a XDL generator. "
                    "Given task description, invalid XDL, and verifier errors, "
                    "return a JSON list of concise syntax/parameter validity constraints."
                ),
            },
            {"role": "user", "content": payload},
        ],
        temperature=0,
        max_tokens=300,
    )
    content = response["choices"][0]["message"]["content"].strip()
    try:
        constraints = json.loads(content)
    except json.JSONDecodeError:
        constraints = [line.strip("- ") for line in content.split("\n") if line.strip()]
    return [item for item in constraints if isinstance(item, str)]


def generate_xdl_with_constraints(description, constraints_text):
    xdl_definition = open("XDL_description.txt", "r").read()
    full_prompt = description
    if constraints_text:
        full_prompt = f"{description}\n\nConstraint rules:\n{constraints_text}"
    output = prompt(full_prompt, xdl_definition, 1000)
    if "<XDL>" not in output:
        return output
    return output[output.index("<XDL>"):]


def run_constraint_bootstrap(
    input_dir,
    rounds,
    stable_window,
    stable_tolerance,
    min_rounds,
    output_path,
):
    samples = load_training_samples(input_dir)
    trainer = ConstraintBootstrapTrainer(
        generate_xdl_fn=generate_xdl_with_constraints,
        suggest_constraints_fn=suggest_constraints_from_failure,
    )
    result = trainer.train(
        samples=samples,
        rounds=rounds,
        stable_window=stable_window,
        stable_tolerance=stable_tolerance,
        min_rounds=min_rounds,
    )
    with open(output_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Constraint bootstrap result saved to {output_path}")
    if result["constraints"]:
        print("Learned constraints:")
        for rule in result["constraints"]:
            print(f"- {rule}")
    print("Accuracy history:", result["accuracy_history"])


def main():
    """main."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--avail_hardware", default=None, type=str)
    parser.add_argument("--avail_reagents", default=None, type=str)
    parser.add_argument("--bootstrap_constraints", action="store_true")
    parser.add_argument("--bootstrap_rounds", default=10, type=int)
    parser.add_argument("--stable_window", default=3, type=int)
    parser.add_argument("--stable_tolerance", default=0.01, type=float)
    parser.add_argument("--min_rounds", default=3, type=int)
    parser.add_argument("--bootstrap_output", default="constraint_bootstrap_result.json", type=str)
    args = parser.parse_args()
    if args.input_dir[-1] == "/":
        args.input_dir = args.input_dir[:-1]
    output_dir = args.input_dir + "_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    available_hardware=None
    # if passed in avail hardware file, parse into list
    if args.avail_hardware != None:
        with open(args.avail_hardware) as f:
            available_hardware = f.read().split("\n")
    print("available hardware:", available_hardware)

    available_reagents=None
    # if passed in avail reagents file, parse into list
    if args.avail_reagents != None:
        with open(args.avail_reagents) as f:
            available_reagents= f.read().split("\n")
    print("available reagents:", available_reagents)

    if args.bootstrap_constraints:
        run_constraint_bootstrap(
            input_dir=args.input_dir,
            rounds=args.bootstrap_rounds,
            stable_window=args.stable_window,
            stable_tolerance=args.stable_tolerance,
            min_rounds=args.min_rounds,
            output_path=args.bootstrap_output,
        )
        return

    num_correct = 0
    total_num = 0
    for rootdir, subdirs, filenames in os.walk(args.input_dir):
        for ii, filename in tqdm(enumerate(sorted(filenames))):
            print(filename)
            if os.path.exists(os.path.join(output_dir, filename)):
                continue
            if ".txt" not in filename:
                continue
            try:
                correct_syntax, xdl, errors = generate_xdl(
                    os.path.join(rootdir, filename), available_hardware, available_reagents
                )
                print(filename, correct_syntax)
                with open(os.path.join(output_dir, filename), "w") as f:
                    f.write(xdl)
                with open(
                    os.path.join(output_dir, filename.replace(
                        ".txt", "_errors.json")),
                    "w",
                ) as f:
                    json.dump(errors, f)
                total_num += 1
                num_correct += correct_syntax
            except:
                print(filename, "error")
                continue
    print(f"Total num correct:: {num_correct}")
    print(f"Total num:: {total_num}")


if __name__ == "__main__":
    main()
