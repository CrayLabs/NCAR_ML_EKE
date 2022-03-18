"""
Driver script for the execution of SmartSim enabled
MOM6 simulations with the OM4_025 configuration.

To run the exact same experiment as our paper, increase
the time in both batch jobs and the number of days
to 10 years.
"""

from glob import glob
from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.log import log_to_file


def create_mom_ensemble( experiment, args ):
    """Creates the ensemble object that stores all configuration options
    """

    mom6_batch_args = {
        "C":args.ensemble_node_features,
        "exclusive": None
    }

    ensemble_batch_settings = experiment.create_batch_settings(
        nodes=args.ensemble_size*args.nodes_per_member,
        time="1:00:00",
        batch_args = mom6_batch_args
    )

    mom6_run_args = {
        "nodes"  : args.nodes_per_member,
        "ntasks" : args.nodes_per_member*args.ppn_per_member,
        "exclusive": None
    }

    mom6_run_settings = experiment.create_run_settings(
        args.mom6_exe_path,
        run_command = "srun",
        run_args = mom6_run_args
    )

    mom_ensemble = experiment.create_ensemble(
        "MOM",
        batch_settings = ensemble_batch_settings,
        run_settings   = mom6_run_settings,
        replicas       = args.ensemble_size
    )

    mom_ensemble.attach_generator_files(
        to_configure=glob("../MOM6_config/configurable_files/*"),
        to_copy="../MOM6_config/OM4_025",
        to_symlink="/lus/cls01029/shao/dev/gfdl/MOM6-examples/ice_ocean_SIS2/OM4_025/INPUT"
    )

    return mom_ensemble

def configure_mom_ensemble( ensemble, args, colocated_orchestrator, db_nodes ):
    """Configures all the members of the MOM6 ensemble with runtime configureable parameters
    """

    MOM6_config_options = {
        "SIM_DAYS": 1, # length of simlations
        "EKE_MODEL": args.eke_model_name,
        "EKE_BACKEND": args.eke_backend,
        "DOMAIN_LAYOUT": args.domain_layout,
        "MASKTABLE": args.mask_table
    }
    # For now the next entries have to be there to avoid the generator bug
    MOM6_config_options.update( {
        "SMARTREDIS_COLOCATED":"False",
        "SMARTREDIS_COLOCATED_STRIDE":0,
        "SMARTREDIS_CLUSTER": "False"
    })

    if colocated_orchestrator:
        MOM6_config_options.update( {
        "SMARTREDIS_COLOCATED":"True",
        "SMARTREDIS_COLOCATED_STRIDE":18,
        })
    elif db_nodes >= 3:
        MOM6_config_options.update( {'SMARTREDIS_CLUSTER':'True'} )

    for model in ensemble:
        model.params = MOM6_config_options
        model.register_incoming_entity(model)

    return ensemble

def add_colocated_orchestrator( ensemble, args ):
    for model in ensemble:
        model.colocate_db(
            args.orchestrator_port,
            ifname=args.orchestrator_interface,
            limit_app_cpus=False
        )

def create_distributed_orchestrator( args ):
    orchestrator = Orchestrator(
        port = args.orchestrator_port,
        interface = args.orchestrator_interface,
        db_nodes = args.orchestrator_nodes,
        time="1:00:00",
        threads_per_queue=4,
        launcher='slurm',
        batch=True)
    orchestrator.set_cpus(18)
    orchestrator.set_batch_arg("constraint", args.orchestrator_node_features)
    orchestrator.set_batch_arg("exclusive",None)
    return orchestrator

def driver( ensemble_args, orchestrator_args, mom6_args):
    log_to_file("./driver.log", log_level='developer')

    experiment = Experiment("AI-EKE-MOM6", launcher="slurm")
    mom_ensemble = create_mom_ensemble( experiment, ensemble_args )
    configure_mom_ensemble(
        mom_ensemble,
        mom6_args,
        orchestrator_args.colocated_orchestrator,
        orchestrator_args.orchestrator_nodes
    )

    experiment_entities = [ mom_ensemble ]

    if args.colocated_orchestrator:
        add_colocated_orchestrator(mom_ensemble, orchestrator_args)
    else:
        orchestrator = create_distributed_orchestrator(orchestrator_args)
        experiment_entities.append(orchestrator)

    experiment.generate( *experiment_entities, overwrite=True )
    experiment.start( *experiment_entities, summary=True )
    print(experiment.summary())
    experiment.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description= (
            "Run MOM6 online inference example using the Slurm launcher. "
            "The default options will run 12 database nodes with 2432 ranks "
            "for MOM6."),
        add_help=False,
        epilog=(
            "To run in co-located mode:\n"
            "python driver.py --colocated_orchestrator --ensemble_node_features='P100'  "
            "--ppn_per_member=18 --nodes_per_member=16 --orchestrator_interface='lo' "
            "--mask_table 'MOM_mask_table' --domain_layout '16,18'"
        )
    )
    all_args = set()

    ensemble_group = parser.add_argument_group("Ensemble-related settings")
    # Ensemble related settings
    ensemble_group.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of ensemble members"
    )
    ensemble_group.add_argument(
        "--nodes_per_member",
        type=int,
        default=12,
        help="Number of nodes for each ensemble member"
    )
    ensemble_group.add_argument(
        "--ppn_per_member",
        type=int,
        default=40,
        help="Number of processors per node for each ensemble member"
    )
    ensemble_group.add_argument(
        "--mom6_exe_path",
        default="/lus/cls01029/shao/dev/gfdl/MOM6-examples/build/gnu/ice_ocean_SIS2/repro/MOM6",
        type=str,
        help="Location of the MOM6 executable"
    )
    ensemble_group.add_argument(
        "--ensemble_node_features",
        type=str,
        default='CL48',
        help=(
            "The node features requested for the simulation model. "
            "Follows the slurm convention for specifying constraints"
        )
    )
    ensemble_arg_names = set(vars(parser.parse_known_args()[0]).keys()) - all_args
    all_args = all_args.union(ensemble_arg_names)

    orchestrator_group = parser.add_argument_group("Orchestrator-related settings")
    orchestrator_group.add_argument("--orchestrator_nodes",
                        type=int,
                        default=3,
                        help="Number of nodes for the database"
    )
    orchestrator_group.add_argument("--orchestrator_port",
                        type=int,
                        default=6780,
                        help="Port for the database"
    )
    orchestrator_group.add_argument("--orchestrator_interface",
                        type=str,
                        default="ipogif0",
                        help="Network interface for the database"
    )
    orchestrator_group.add_argument("--colocated_orchestrator",
                        action='store_true',
                        dest='colocated_orchestrator',
                        help="If present, run the orchestrator in co-located mode"
    )
    orchestrator_group.add_argument(
        "--orchestrator_node_features",
        type=str,
        default='P100',
        help=(
            "The node features requested for the orchestrator. "
            "Follows the slurm convention for specifying constraints"
        )
    )
    orchestrator_arg_names = set(vars(parser.parse_known_args()[0]).keys()) - all_args
    all_args = all_args.union(orchestrator_arg_names)

    mom6_group = parser.add_argument_group("MOM6-related settings")
    # MOM6 Configuration options
    mom6_group.add_argument(
        "--mask_table",
        type=str,
        default="mask_table.808.45x72",
        help=(
            "The mask table describing the land processor elimination for the "
            "given layout"
        )
    )
    mom6_group.add_argument(
        "--domain_layout",
        type=str,
        default="45,72",
        help=(
            "The domain decomposition to use for MOM6. This must be consistent "
            "with mask_table"
        )
    )
    mom6_group.add_argument(
        "--eke_model_name",
        type=str,
        default="ncar_ml_eke.gpu.pt",
        help="The trained machine learning model used for inferring EKE"
    )
    mom6_group.add_argument(
        "--eke_backend",
        type=str,
        default="GPU",
        help="GPU or CPU, the type of device the inference will be done on"
    )
    mom6_arg_names = set(vars(parser.parse_known_args()[0]).keys()) - all_args
    all_args = all_args.union(mom6_arg_names)

    parser.add_argument("-h", "--help",
       action="help",
       help="show this help message and exit"
    )
    args = parser.parse_args()

    # Parse each of the option groups
    ensemble_args, orchestrator_args, mom6_args = (
        argparse.Namespace(**dict((k,v) for k,v in vars(args).items() if k in arg_group))
        for arg_group in [ensemble_arg_names, orchestrator_arg_names, mom6_arg_names])

    driver( ensemble_args, orchestrator_args, mom6_args )



