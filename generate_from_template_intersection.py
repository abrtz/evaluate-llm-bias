import pandas as pd
import io
import json
from utils import *
import random
import re
import ast
import os
import argparse

def create_dataset_intersection(lang,cat,control):
    """Create and write a dataset from a template with intersection. Datasets are saved under /data directory.
    
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
        words = vocab[vocab.Category == ("Nationality_es")]

        msc_nationalities = words[(words.SubCat == "hombre") | (words.SubCat.isnull())]
        msc_nationalities =  msc_nationalities.Name.unique().tolist()
        fem_nationalities = words[(words.SubCat == "mujer") | (words.SubCat.isnull())] 
        fem_nationalities = fem_nationalities.Name.unique().tolist()

        # make sure that one of the words used is the target of the bias
        possible_word_list = words.Name.unique().tolist()
        bias_targets = the_frames.Known_stereotyped_nationality[i]

        #  Stereotyped groups info not relevant for subsetting Names in some categories
        if len(bias_targets) > 0:
            bias_targets = ast.literal_eval(the_frames.Known_stereotyped_nationality[i])
            targeted_word_list = [x for x in possible_word_list if x in bias_targets]
        else:
            targeted_word_list = possible_word_list
            bias_targets = []

        gender_instance = "hombre" 
        for item in bias_targets:
            if item in fem_nationalities:
                gender_instance = "mujer"
                break
        
        if len(targeted_word_list) > 4:
            word_list = random.sample(targeted_word_list, 5)
        else:
            word_list = targeted_word_list
    
        # set really helpful boolean
        # This determines if the NAME1 and NAME2 needs to be a proper name or not = in case of control template
        if "Proper_nouns_only" in frame_cols:
            if str(the_frames.Proper_nouns_only[i]) == "True":
                has_proper_name = True
            else:
                has_proper_name = False
        else:
            has_proper_name = False

        
        # get the bias target of the second variable (gender)
        target_var2 = the_frames.Known_stereotyped_var2[i]

        # print(f"target: {target_var2}")

        # get info from the Names column
        critical_words = the_frames.Names[i]
        if len(critical_words) > 1:
            critical_words = ast.literal_eval(critical_words)
        
        # get corresponding info about the words
        critical_words_info = the_frames.Var2_info[i]
        if len(critical_words_info) > 1:
            critical_words_info = ast.literal_eval(critical_words_info)

        # Nationality_x_Gender normal always has Names
        if not has_proper_name or the_frames.Names[
            i
        ] != "":
            t_loc = critical_words_info.index(target_var2)
            t_word_gs = critical_words[t_loc]
            non_t_word_gs = critical_words[
                1 - t_loc
            ]  # there's only ever 2 possibilities, take the other one


        # check if this one needs to use proper names
        if has_proper_name:
            # grab possible first names
            # further pare down based on gender when needed
            first_names = names_vocab[names_vocab.gender == target_var2]

            word_list = first_names.name.tolist()
            word_list = random.sample(word_list, 5)  # for downsampling

        # print(F"word list: {word_list}")

        # iterate over each word in word_list
        for j in range(len(word_list)):
            # initialize variables for later
            Name1_info = None
            Name2_info = None
            rand_wrd1 = ""
            rand_wrd2 = ""
            Name2_info_dict = {}

            # current target word
            t_word = word_list[j]

            # create new word list

            #filtering for the nationalities that are not of the same gender
            if gender_instance == "mujer":
                new_word_list = [n for n in msc_nationalities if n not in word_list]
            
            elif gender_instance == "hombre":
                new_word_list = [n for n in fem_nationalities if n not in word_list]

            new_word_list = random.sample(new_word_list, 5)  # downsample

            # getting proper names
            if has_proper_name:
                # get target name info
                the_word_gender = first_names.loc[
                    first_names["name"] == t_word, "gender"
                ].iloc[0]
                Name1_info = the_word_gender

                # create lists of non-target names
                other_first_names = names_vocab[
                names_vocab.gender != the_word_gender
                ].reset_index()
                other_first_names = other_first_names.sample(
                    n=5, replace=False
                ).reset_index()  # for downsampling

                # initialize empty list and dict to hold new names and info about them
                new_word_list = []
                Name2_info_dict = {}
                for nam in range(len(other_first_names)):
                    frst = other_first_names.name[nam]
                    Name2_info = other_first_names.gender[nam]
                    
                    Name2_info_dict[frst] = Name2_info

                    new_word_list.append(frst)

            # print(f"new word list: {new_word_list}")
        
            name1_info_static = Name1_info


            # iterate over each word in the second word list
            # results in full pairings of every possible vocab pair
            for k in range(len(new_word_list)):
                word2 = new_word_list[k]

                if not control:

                    # get word or word subset
                    if isinstance(t_word_gs, list) and len(t_word_gs) > 0:
                        t_word_gs = random.choice(t_word_gs)
                        non_t_word_gs = random.choice(non_t_word_gs)

                this_frame_row = the_frames.iloc[[i]].reset_index()
                lex_div = this_frame_row.Lexical_diversity[0]

                # Only need to create these values when there's text in lexical diversity
                if len(lex_div) > 1:
                    wrdlist1, wrdlist2 = return_list_from_string(lex_div)
                    rand_wrd1 = random.choice(wrdlist1)
                    if len(wrdlist2) > 1:  # sometimes there's not a WORD2
                        rand_wrd2 = random.choice(wrdlist2)
            
                # need to record info about the names that were used for easier analysis later
                if Name1_info is not None and has_proper_name:
                    Name2_info = Name2_info_dict[word2]
                # covers all Race_x_gender and Race_x_SES cases where Names is specified
                elif this_frame_row.Var2_info[0] != "":
                    if target_var2 == "male":
                        nontarget_var2 = "female"
                    elif target_var2 == "female":
                        nontarget_var2 = "male"
                    Name1_info = target_var2 + "-" + t_word  # add gender/nationality info
                    Name2_info = nontarget_var2 + "-" + word2  # add gender/nationality info
                
                # print(Name1_info)
                # print(Name2_info)
                # print()

                # replace frame text info with value of {{NAME}} and {{WORD}}.
                # then set everything into the formatting needed to save the data
                # then repeat with names in the reverse order
                # this has to be done slightly differently depending on which category is being used and whether
                # the template uses a proper name. No matter what, there's just a handful of fiddly differences
                # in the way the names are stored, so each one is handled a little differently using the same functions,
                # but this gets very repetitive.
                if not has_proper_name:
                    new_frame_row = do_slotting_intersection(
                        this_frame_row,
                        frame_cols,
                        t_word_gs, #in ES the gender noun comes before the nationality adj (in all cases)
                        t_word,
                        non_t_word_gs,
                        word2,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )

                    # create four sets of data, each as a dictionary
                    dat_formatted = create_templating_dicts(
                        cat,
                        new_frame_row,
                        "intersectional",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [t_word_gs, t_word],
                        [non_t_word_gs, word2],
                        Name1_info,
                        Name2_info,
                        nn,
                    )
                    nn += 4

                    # now reverse everything for counterbalancing
                    new_frame_row_rev = do_slotting_intersection(
                        this_frame_row,
                        frame_cols,
                        non_t_word_gs, #in ES the gender noun comes before the nationality adj
                        word2,
                        t_word_gs,
                        t_word,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    
                    # create four sets of data, each as a dictionary
                    dat_formatted_rev = create_templating_dicts(
                        cat,
                        new_frame_row_rev,
                        "intersectional",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        [non_t_word_gs, word2],
                        [t_word_gs, t_word],
                        Name2_info,
                        Name1_info,
                        nn,
                    )
                    nn += 4

                if has_proper_name:
                    new_frame_row = do_slotting(
                        this_frame_row,
                        frame_cols,
                        t_word,
                        None,
                        word2,
                        None,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                    
                    # create four sets of data, each as a dictionary
                    dat_formatted = create_templating_dicts(
                        cat,
                        new_frame_row,
                        "intersectional",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        t_word,
                        word2,
                        Name1_info,
                        Name2_info,
                        nn,
                    )
                    nn += 4       

                    # now reverse them all for counterbalancing
                    new_frame_row_rev = do_slotting(
                        this_frame_row,
                        frame_cols,
                        word2,
                        None,
                        t_word,
                        None,
                        lex_div,
                        rand_wrd1,
                        rand_wrd2,
                    )
                        
                    # create four sets of data, each as a dictionary
                    dat_formatted_rev = create_templating_dicts(
                        cat,
                        new_frame_row_rev,
                        "intersectional",
                        unknown_options,
                        frame_cols,
                        bias_targets,
                        word2,
                        t_word,
                        Name2_info,
                        Name1_info,
                        nn,
                    )
                    nn += 4
                    
                for d_formatted in [
                    dat_formatted,
                    dat_formatted_rev
                ]:
                    for item in d_formatted:
                        dat_file.write(json.dumps(item, default=str, ensure_ascii=False))
                        dat_file.write("\n")
                    dat_file.flush()

        print("generated %s sentences total for %s" % (str(nn), cat) + (" control" if control else ""))

    dat_file.close()


def main():

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Create datasets from templates.")
    parser.add_argument('--lang', type=str, default="es", help="Language code to build dataset (default: 'es')")
    parser.add_argument('--cat', type=str, default="Gender_x_nationality", help="Bias category name (default: 'Gender_x_nationality')")
    parser.add_argument("--control", type=bool, default=False, help="Build dataset based on Control template (default: False)")
    args = parser.parse_args()

    print(f"Creating dataset with langauge:{args.lang}, category:{args.cat}, control:{args.control}")

    create_dataset_intersection(lang=args.lang, cat=args.cat, control=args.control)

# Run main function when called from CLI
if __name__ == "__main__":
    main()


