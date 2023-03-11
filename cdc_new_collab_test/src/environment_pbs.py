"""Configuration of the project enviroment. 
 
The environments defined in this module can be auto-detected. 
This helps to define environment specific behaviour in heterogenous 
environments. 
""" 
import flow 
 
__all__ = ['CoriEnvironment']
 
class RahmanEnvironment(flow.environment.DefaultPBSEnvironment):
    
    template = 'pbs.sh'

    @classmethod
    def add_args(cls, parser):
        super(flow.environment.DefaultPBSEnvironment, cls).add_args(parser)
        parser.add_argument(
               '-w', "--walltime", type=float, help="Walltime"
            )
