from argparse import Namespace

from utils.orchestrator_argument_utils import build_base_argument_parser, expand_argument_parser_with_ae_pretraining_parameters, \
    set_env_variables, namespace_to_global_args_and_search_space


def setup_arguments() -> dict:
    parser = build_base_argument_parser()
    parser = expand_argument_parser_with_ae_pretraining_parameters(parser)

    args: Namespace = parser.parse_args()

    set_env_variables(args)

    return namespace_to_global_args_and_search_space(args)

if __name__ == "__main__":
    global_args, search_space_args = setup_arguments()
    print("Configuration:", global_args)
    print("Search Space:", search_space_args)
