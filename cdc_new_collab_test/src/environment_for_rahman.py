"""Configuration of the project enviroment. 
 
The environments defined in this module can be auto-detected. 
This helps to define environment specific behaviour in heterogenous 
environments. 
""" 
import flow 
 
__all__ = ['CoriEnvironment']
 
class CoriEnvironment(flow.environment.DefaultSlurmEnvironment): 
    # template = 'cori_requeue.sh'
    # template = 'perl.sh'
    template = 'perl_regular_requeue.sh'
    # template = 'perl_preempt_requeue.sh'

    @classmethod
    def add_args(cls, parser):
        super(flow.environment.DefaultSlurmEnvironment, cls).add_args(parser)
        parser.add_argument(
               '-w', "--walltime", type=float, help="Walltime"
            )
