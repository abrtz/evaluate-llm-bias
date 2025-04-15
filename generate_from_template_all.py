import pandas as pd
import io
import json
from utils import *
import random
import ast
import os
import argparse


def create_dataset(lang,cat,control):
    """Create and write a dataset from a template. Datasets are saved under /data directory.
    
    Parameters:
    - lang (str): Language code to build dataset.
    - cat (str): Bias category name.
    - control (bool): Build dataset based on Control template.

    Return None.
    """

    # read in vocabulary files
    vocab = pd.read_csv("templates/vocabulary_with_es.csv", sep=";") 
    vocab = vocab[vocab.Pilot_include != "No"]
    names_vocab = pd.read_csv("templates/top_names.csv") #if control else "BBQ-main/templates/vocabulary_proper_names.csv")

    unknown_options = [
        "No se sabe",
        "No se puede responder",
        "No puedo responder",
        "No se puede determinar"
    ]

    frame_path = (
        f"templates/control_templates - {cat}_{lang}.csv" if control else f"templates/new_templates - {cat}_{lang}.csv"
    ) # get the right template file

    frames = pd.read_csv(frame_path, sep=";", na_filter=False) 

    os.makedirs(os.path.dirname(f"data/{cat}{'_control' if control else ''}_{lang}.jsonl"), exist_ok=True) #create the file if it doesn't exist already

    dat_file = io.open(
            f"data/{cat}{'_control' if control else ''}_{lang}.jsonl", "w", encoding="utf-8"
        )  # open the file that all the data will be saved in

    the_frames = frames[
        frames.Ambiguous_Context != ""
    ].reset_index()  # get rid of ones that are empty
    frame_cols = the_frames.columns

    # initialize item id counter
    nn = 0

    for i in range(len(the_frames)):

        # by default, take the words from the vocab that match this category
        # this needs to be done at each iteration because some rows use custom vocab and some don't
        this_subcat = the_frames.Stated_gender_info[i] #dividing the nationalities by gender
        names = names_vocab[names_vocab.gender == this_subcat] ###keep the fem or msc names

        words = vocab[vocab.Category == (cat+"_es")]
        words = words[(words.SubCat == this_subcat) | (words.SubCat.isnull())] ###keep the femenine or masculine nationality adj as well as those without SubCat
        
        # make sure that one of the words used is the target of the bias
        possible_word_list = words.Name.unique().tolist()
        bias_targets = the_frames.Known_stereotyped_groups[i]

        # Stereotyped groups info is not relevant for subsetting Names in some categories
        # so only use that for creating the targeted word list when the info is provided, otherwise all possible
        # vocab items are targets
        if len(bias_targets) > 1: #and (cat in need_stereotyping_subset):
            bias_targets = ast.literal_eval(the_frames.Known_stereotyped_groups[i]) ###convert to native python object, in this case, list of strings
            targeted_word_list = [x for x in possible_word_list if x in bias_targets]
            
        else: 
            targeted_word_list = possible_word_list
            bias_targets = ast.literal_eval(the_frames.Known_stereotyped_groups[i])


        if control:

            # set really helpful boolean
            # This determines if the NAME1 and NAME2 needs to be a proper name or not
            if "Proper_nouns_only" in frame_cols:
                if str(the_frames.Proper_nouns_only[i]) == "True":
                    has_proper_name = True
                else:
                    has_proper_name = False
            else:
                has_proper_name = False

            # check if this one needs to use proper names - for CONTROL
            if has_proper_name:
                word_list = names.name.tolist()
                word_list = random.sample(word_list, 5)  # for downsampling

        else:
        
            # if the list of bias targets is too big, downsample
            if len(targeted_word_list) > 4:
                word_list = random.sample(targeted_word_list, 5)
            elif len(possible_word_list) < 2:  # these will be handled later
                word_list = []
            else: ### cases in which the list smaller or equal to 4 items
                word_list = targeted_word_list


        critical_words = ""
        
        #print(f"List of words: {word_list}")

        # iterate over each word in word_list
        for j in range(len(word_list)):
            # initialize variables for later
            Name1_info = None
            Name2_info = None
            rand_wrd1 = ""
            rand_wrd2 = ""
            Name2_info_dict = {}

            # current target word
            this_word = word_list[j]
            #print(this_word)

            # only create new_word_list here if it wasn't already created through Names column
            if len(critical_words) < 2:

                if control:
                    the_word_gender = names.loc[names["name"] == this_word, "gender"].iloc[0] ### get the gender of "this_word" (current name)
                    Name1_info = the_word_gender
                    #new_options = names[names["name"] != this_word]
                    new_options = names[[name not in word_list for name in names["name"]]] #exclude the names in the first list
                    new_word_list = new_options.name.unique().tolist()
                    new_word_list = random.sample(new_word_list, 5)  # for downsampling

                else:
                    new_word_list = [
                        n for n in possible_word_list if n not in bias_targets
                        ]
                
                    if len(new_word_list) > 4:
                        new_word_list = random.sample(
                            new_word_list, 5
                        )  # downsample when needed
                    else:
                        new_word_list = [n for n in word_list if n != this_word]

            #print(f"Second list of words: {new_word_list}")
            #print()

            # iterate over each word in the second word list
            # results in full pairings of every possible vocab pair within the subset
            for k in range(len(new_word_list)):
                this_word_2 = new_word_list[k]
                this_frame_row = the_frames.iloc[[i]].reset_index()
                lex_div = this_frame_row.iloc[0, 7] ### Lexical_diversity is the located in the 7th column of the frame's row

                # Only need to create these values when there's text in lexical diversity
                if len(lex_div) > 1:
                    wrdlist1, wrdlist2 = return_list_from_string(lex_div)
                    rand_wrd1 = random.choice(wrdlist1)
                    if len(wrdlist2) > 1:  # sometimes there's not a WORD2
                        rand_wrd2 = random.choice(wrdlist2)

                # replace frame text info with value of {{NAME}} and {{WORD}}. Check the value for each column
                new_frame_row = do_slotting(
                    this_frame_row,
                    frame_cols,
                    this_word,
                    None,
                    this_word_2,
                    None,
                    lex_div,
                    rand_wrd1,
                    rand_wrd2,
                )

                if control:
                    # need to record info about the names that were used for easier analysis later
                    #Name1_info already retrieved above
                    #assign the opposite gender to Name2_info so there's always a contrast
                    Name2_info = "male" if Name1_info == "female" else "female"

                else:
                    # need to get the relevant info about the name from the vocab file
                    Name1_info = vocab.loc[vocab["Name"] == this_word, "Info"].iloc[0]
                    Name2_info = vocab.loc[vocab["Name"] == this_word_2, "Info"].iloc[0]

                # create four sets of data, each as a dictionary
                dat_formatted = create_templating_dicts(
                    #cat.replace("_es",""),
                    cat,
                    new_frame_row,
                    this_subcat,
                    unknown_options,
                    frame_cols,
                    bias_targets,
                    this_word,
                    this_word_2,
                    Name1_info,
                    Name2_info,
                    nn,
                )

                nn += 4
                for item in dat_formatted:
                    dat_file.write(json.dumps(item, default=str, ensure_ascii=False))
                    dat_file.write("\n")
                dat_file.flush()


                ## because category needs stereotyping subset:
                # flip input of this_word and this_word_2
                new_frame_row = do_slotting(
                    this_frame_row,
                    frame_cols,
                    this_word_2,
                    None,
                    this_word,
                    None,
                    lex_div,
                    rand_wrd1,
                    rand_wrd2,
                )

                # create four sets of data, each as a dictionary
                dat_formatted = create_templating_dicts(
                    #cat.replace("_es",""),
                    cat,
                    new_frame_row,
                    this_subcat,
                    unknown_options,
                    frame_cols,
                    bias_targets,
                    this_word_2,
                    this_word,
                    Name2_info,
                    Name1_info,
                    nn,
                )
                nn += 4
                for item in dat_formatted:
                    dat_file.write(json.dumps(item, default=str, ensure_ascii=False))
                    dat_file.write("\n")
                dat_file.flush()

        print(("generated %s sentences total for %s" % (str(nn), cat)) + (" control" if control else ""))

    dat_file.close()

def main():

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Create datasets from templates.")
    parser.add_argument('--lang', type=str, default="es", help="Language code to build dataset (default: 'es')")
    parser.add_argument('--cat', type=str, default="Nationality", help="Bias category name (default: 'Nationality')")
    parser.add_argument("--control", type=bool, default=False, help="Build dataset based on Control template (default: False)")
    args = parser.parse_args()

    print(f"Creating dataset with langauge:{args.lang}, category:{args.cat}, control:{args.control}")

    create_dataset(lang=args.lang, cat=args.cat, control=args.control)

# Run main function when called from CLI
if __name__ == "__main__":
    main()

