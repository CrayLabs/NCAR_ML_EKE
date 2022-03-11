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


def create_mom_ensemble( experiment, mom6_exe_path, ensemble_size, nodes_per_member, constraints ):
    """Creates the ensemble object that stores all configuration options
    """

    mom6_batch_args = {
        "C":constraints,
        "exclusive": None
    }

    ensemble_batch_settings = experiment.create_batch_settings(
        nodes=ensemble_size*nodes_per_member,
        time="1:00:00",
        batch_args = mom6_batch_args
    )

    mom6_run_args = {
        "nodes"  : nodes_per_member,
        "ntasks" : 480, # This number must match the ranks in MASKTABLE below
        "exclusive": None
    }

    mom6_run_settings = experiment.create_run_settings(
        mom6_exe_path,
        run_command = "srun",
        run_args = mom6_run_args
    )

    mom_ensemble = experiment.create_ensemble(
        "MOM",
        batch_settings = ensemble_batch_settings,
        run_settings   = mom6_run_settings,
        replicas       = ensemble_size
    )

    mom_ensemble.attach_generator_files(
        to_configure=glob("../MOM6/MOM6_config/*"),
        to_copy="../MOM6/OM4_025",
        to_symlink="/lus/cls01029/shao/dev/gfdl/MOM6-examples/ice_ocean_SIS2/OM4_025/INPUT"
    )

    return mom_ensemble

def configure_mom_ensemble( ensemble ):
    MOM6_config_options = {
        "SIM_DAYS": 1, # length of simlations
        "DOMAIN_LAYOUT": "32,18",
        "MASKTABLE": "mask_table.96.32x18"
    }
    for model in ensemble:
        model.params = MOM6_config_options

    return ensemble

def add_colocated_orchestrator( ensemble, port, interface ):
    for model in ensemble:
        model.colocate_db(port, ifname=interface)

def create_distributed_orchestrator( nodes, port, interface, node_features ):
    orchestrator = Orchestrator(
        port = port,
        interface = interface,
        db_nodes=nodes,
        time="1:00:00",
        threads_per_queue=4,
        launcher='slurm',
        batch=True)
    orchestrator.set_cpus(36)
    orchestrator.set_batch_arg("constraint",node_features)
    orchestrator.set_batch_arg("exclusive",None)
    return orchestrator

def driver( args ):
    log_to_file("./driver.log")

    experiment = Experiment("AI-EKE-MOM6", launcher="slurm")
    mom_ensemble = create_mom_ensemble(
        experiment,
        args.mom6_exe_path,
        args.ensemble_size,
        args.nodes_per_member,
        args.ensemble_node_features
    )
    configure_mom_ensemble(mom_ensemble)

    experiment_entities = [ mom_ensemble ] # This list holds all the entities to start

    if args.colocated_orchestrator:
        add_colocated_orchestrator(mom_ensemble)
    else:
        orchestrator = create_distributed_orchestrator(
            args.orchestrator_nodes,
            args.orchestrator_port,
            args.orchestrator_interface,
            args.orchestrator_node_features
        )
        experiment_entities.append(orchestrator)

    experiment.generate( *experiment_entities, overwrite=True )
    experiment.start( *experiment_entities, summary=True )
    print(experiment.summary())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MOM6 online inference example using the Slurm launcher")
    # MOM6 related settings
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of ensemble members"
    )
    parser.add_argument(
        "--nodes_per_member",
        type=int,
        default=14,
        help="Number of nodes for each ensemble member"
    )
    parser.add_argument(
        "--ppn_per_member",
        type=int,
        default=36,
        help="Number of processors per node for each ensemble member"
    )
    parser.add_argument(
        "--mom6_exe_path",
        type=str,
        default="/lus/cls01029/spartee/poseidon/NCAR_ML_EKE/MOM6/build/gnu/ice_ocean_SIS2/repro/MOM6",
        help="Location of the MOM6 executable"
    )
    parser.add_argument(
        "--ensemble_node_features",
        type=str,
        default='CL48',
        help="The node features requested for the simulation model. Follows the slurm convention for specifying constraints"
    )

    # Orchestrator options
    parser.add_argument("--orchestrator_nodes",
                        type=int,
                        default=3,
                        help="Number of nodes for the database"
    )
    parser.add_argument("--orchestrator_port",
                        type=int,
                        default=6780,
                        help="Port for the database"
    )
    parser.add_argument("--orchestrator_interface",
                        type=str,
                        default="ipogif0",
                        help="Network interface for the database"
    )
    parser.add_argument("--colocated_orchestrator",
                        action='store_true',
                        dest='colocated_orchestrator',
                        help="If present, run the orchestrator in co-located mode"
    )
    parser.add_argument(
        "--orchestrator_node_features",
        type=str,
        default='P100',
        help="The node features requested for the orchestrator. Follows the slurm convention for specifying constraints"
    )
    args = parser.parse_args()

    driver( args )



