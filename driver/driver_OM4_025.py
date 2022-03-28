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
import warnings

def create_mom_ensemble(
        experiment,
        walltime,
        ensemble_size,
        nodes_per_member,
        tasks_per_node,
        mom6_exe_path,
        ensemble_node_features=None
    ):

    if experiment._launcher == 'slurm' and ensemble_node_features:
        mom6_batch_args = {
            "C":ensemble_node_features,
        }
    else:
        warnings.warn(
            "ensemble_node_features was set, but the launcher is not slurm."+
            "Ignoring this option"
        )

        mom6_batch_args = None

    ensemble_batch_settings = experiment.create_batch_settings(
        nodes      = ensemble_size*nodes_per_member,
        time       = walltime,
        batch_args = mom6_batch_args
    )

    mom6_run_settings = experiment.create_run_settings(mom6_exe_path)
    mom6_run_settings.set_tasks_per_node(tasks_per_node)
    mom6_run_settings.set_tasks(nodes_per_member*tasks_per_node)

    mom_ensemble = experiment.create_ensemble(
        "MOM",
        batch_settings = ensemble_batch_settings,
        run_settings   = mom6_run_settings,
        replicas       = ensemble_size
    )

    mom_ensemble.attach_generator_files(
        to_configure=glob("../MOM6_config/configurable_files/*"),
        to_copy="../MOM6_config/OM4_025",
        to_symlink="../MOM6_config/INPUT"
    )

    return mom_ensemble

def configure_mom_ensemble(
    ensemble,
    colocated_orchestrator,
    clustered_orchestrator,
    mask_table,
    domain_layout,
    eke_model_name,
    eke_backend,
    colocated_stride=0
    ):

    MOM6_config_options = {
        "SIM_DAYS": 15, # length of simlations
        "EKE_MODEL": eke_model_name,
        "EKE_BACKEND": eke_backend,
        "DOMAIN_LAYOUT": domain_layout,
        "MASKTABLE": mask_table
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
        "SMARTREDIS_COLOCATED_STRIDE":colocated_stride,
        })
    if clustered_orchestrator:
        MOM6_config_options.update( {'SMARTREDIS_CLUSTER':'True'} )

    for model in ensemble:
        model.params = MOM6_config_options
        model.register_incoming_entity(model)

    return ensemble

def add_colocated_orchestrator(
    ensemble,
    orchestrator_port,
    orchestrator_interface,
    orchestrator_cpus,
    limit_app_cpus
    ):

    for model in ensemble:
        model.colocate_db(
            port=orchestrator_port,
            ifname=orchestrator_interface,
            db_cpus=orchestrator_cpus,
            limit_app_cpus=limit_app_cpus
        )

def create_distributed_orchestrator(
    exp,
    orchestrator_port,
    orchestrator_interface,
    orchestrator_nodes,
    orchestrator_node_features,
    walltime
    ):

    orchestrator = exp.create_database(
        port = orchestrator_port,
        interface = orchestrator_interface,
        db_nodes = orchestrator_nodes,
        time=walltime,
        threads_per_queue=2,
        batch=True)

    orchestrator.set_cpus(18)
    orchestrator.set_batch_arg("constraint", orchestrator_node_features)
    orchestrator.set_batch_arg("exclusive",None)
    return orchestrator

def mom6_colocated_driver(
    walltime="02:00:00",
    ensemble_size=1,
    nodes_per_member=15,
    tasks_per_node=17,
    mom6_exe_path="/lus/cls01029/shao/dev/gfdl/MOM6-examples/build/gnu/"+
                  "ice_ocean_SIS2/repro/MOM6",
    ensemble_node_features='P100',
    mask_table="mask_table.33.16x18",
    domain_layout="16,18",
    eke_model_name="ncar_ml_eke.gpu.pt",
    eke_backend="GPU",
    orchestrator_port=6780,
    orchestrator_interface="ipogif0",
    colocated_stride=18,
    orchestrator_cpus=4,
    limit_orchestrator_cpus=False
    ):
    """Run a MOM6 OM4_025 simulation using a colocated deployment for online
    machine-learning inference

    :param walltime: how long to allocate for the run, "hh:mm:ss"
    :type walltime: str, optional
    :param ensemble_size: number of members in the ensemble
    :type ensemble_size: int, optional
    :param nodes_per_member: number of nodes allocated to each ensemble member
    :type nodes_per_member: int, optional
    :param tasks_per_node: how many MPI ranks to be run per node
    :type tasks_per_node: int, optional
    :param mom6_exe_path: full path to the compiled MOM6 executable
    :type mom6_exe_path: str, optional
    :param ensemble_node_features: (Slurm-only) Constraints/features for the
                                    node
    :type ensemble_node_features: str, optional
    :param mask_table: the file to use for the specified layout eliminating
                       land domains
    :type mask_table: str, optional
    :param domain_layout: the particular domain decomposition
    :type domain_layout: str, optional
    :param eke_model_name: file containing the saved machine-learning model
    :type eke_model_name: str, optional
    :param eke_backend: (CPU or GPU), sets whether the ML-EKE model will be
                        run on CPU or GPU
    :type eke_backend: str, optional
    :param orchestrator_port: port that the database will listen on
    :type orchestrator_port: int, optional
    :param orchestrator_interface: network interface bound to the orchestrator
    :type orchestrator_interface: str, optional
    :param orchestrator_cpus: Specify the number of cores that the
                                    orchestrator can use to handle requests
    :type orchestrator_cpus: int, optional
    :param limit_orchestrator_cpus: Limit the number of CPUs that the
                                    orchestrator can use to handle requests
    :type limit_orchestrator_cpus: bool, optional
    """
    experiment = Experiment("AI-EKE-MOM6", launcher="auto")
    mom_ensemble = create_mom_ensemble(
        experiment,
        walltime,
        ensemble_size,
        nodes_per_member,
        tasks_per_node,
        mom6_exe_path,
        ensemble_node_features
    )
    configure_mom_ensemble(
        mom_ensemble,
        True,
        False,
        mask_table,
        domain_layout,
        eke_model_name,
        eke_backend,
        colocated_stride=colocated_stride)

    add_colocated_orchestrator(
        mom_ensemble,
        orchestrator_port,
        orchestrator_interface,
        orchestrator_cpus,
        limit_orchestrator_cpus,
    )

    experiment.generate( mom_ensemble, overwrite=True )
    experiment.start( mom_ensemble, summary=True )
    experiment.stop()

def mom6_clustered_driver(
    walltime="02:00:00",
    ensemble_size=1,
    nodes_per_member=25,
    tasks_per_node=45,
    mom6_exe_path="/lus/cls01029/shao/dev/gfdl/MOM6-examples/build/gnu/"+
                  "ice_ocean_SIS2/repro/MOM6",
    ensemble_node_features='[CL48|SK48|SK56]',
    mask_table="mask_table.315.32x45",
    domain_layout="32,45",
    eke_model_name="ncar_ml_eke.gpu.pt",
    eke_backend="GPU",
    orchestrator_port=6780,
    orchestrator_interface="ipogif0",
    orchestrator_nodes=3,
    orchestrator_node_features='P100',
    configure_only=False
    ):
    """Run a MOM6 OM4_025 simulation with a cluster of databases used for
    machine-learning inference

    :param walltime: how long to allocate for the run, "hh:mm:ss"
    :type walltime: str, optional
    :param ensemble_size: number of members in the ensemble
    :type ensemble_size: int, optional
    :param nodes_per_member: number of nodes allocated to each ensemble member
    :type nodes_per_member: int, optional
    :param tasks_per_node: how many MPI ranks to be run per node
    :type tasks_per_node: int, optional
    :param mom6_exe_path: full path to the compiled MOM6 executable
    :type mom6_exe_path: str, optional
    :param ensemble_node_features: (Slurm-only) Constraints/features for the
                                    node
    :type ensemble_node_features: str, optional
    :param mask_table: the file to use for the specified layout eliminating
                       land domains
    :type mask_table: str, optional
    :param domain_layout: the particular domain decomposition
    :type domain_layout: str, optional
    :param eke_model_name: file containing the saved machine-learning model
    :type eke_model_name: str, optional
    :param eke_backend: (CPU or GPU), sets whether the ML-EKE model will be
                        run on CPU or GPU
    :type eke_backend: str, optional
    :param orchestrator_port: port that the database will listen on
    :type orchestrator_port: int, optional
    :param orchestrator_interface: network interface bound to the database
    :type orchestrator_interface: str, optional
    :param orchestrator_nodes: number of orchestrator nodes to use
    :type orchestrator_nodes: int, optional
    :param orchestrator_node_features: (Slurm-only) node features requested for
                                       the orchestrator nodes
    :type orchestrator_node_features: str, optional
    :param configure_only: If True, only configure the experiment and return
                           the orchestrator and experiment objects
    :type configure_only: bool, optional
    """

    experiment = Experiment("AI-EKE-MOM6", launcher="auto")
    mom_ensemble = create_mom_ensemble(
        experiment,
        walltime,
        ensemble_size,
        nodes_per_member,
        tasks_per_node,
        mom6_exe_path,
        ensemble_node_features
     )
    configure_mom_ensemble(
        mom_ensemble,
        False,
        orchestrator_nodes>=3,
        mask_table,
        domain_layout,
        eke_model_name,
        eke_backend
    )
    orchestrator = create_distributed_orchestrator(
        experiment,
        orchestrator_port,
        orchestrator_interface,
        orchestrator_nodes,
        orchestrator_node_features,
        walltime
    )

    experiment.generate( mom_ensemble, orchestrator, overwrite=True )
    if configure_only:
        return experiment, mom_ensemble, orchestrator
    else:
        experiment.start(mom_ensemble, orchestrator, summary=True)
        experiment.stop(orchestrator)

if __name__ == "__main__":
    import fire
    log_to_file("./mom6_driver.log", log_level='info')
    fire.Fire({
        "colocated":mom6_colocated_driver,
        "clustered":mom6_clustered_driver
    })
