/**
 * Package for fitting Bayesian GLMs - currently logistic and Weibull regression
 * models are available. Approximate Posterior samples are drawn using an MCMC 
 * sampler with a (Reversible Jump) Metropolis-Hastings acceptance ratio.
 * 
 * - Allows fitting of random intercepts
 * - Allows likelihoodFamily selection through a Reversible Jump algorithm
 * 
 * @author Paul J Newcombe
 */
package bglims;

import Methods.GeneralMethods;
import Methods.RegressionMethods;
import Objects.Arguments;
import Objects.Data;
import Objects.IterationValues;
import Objects.LikelihoodTypes;
import Objects.ParameterTypes;
import Objects.Priors;
import Objects.ProposalDistributions;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import umontreal.iro.lecuyer.rng.F2NL607;

/**
 * Contains main method (the constructor) for fitting Bayesian GLMs.
 * 
 * @author Paul J Newcombe
 */
public class BGLiMS {
    /**
     * Random number generator.
     */
    public static Random randomDraws;
    /**
     * For unseeded random number generation.
     */
    public static F2NL607 stream = new F2NL607();
    /**
     * Object of class {@link Objects.ParameterTypes} describing the parameter types.
     */
    public static ParameterTypes ParamTypes;

    /**
     * Constructor - orchestrates the MCMC algorithm.
     * 
     * @param args Command line arguments passed to the program. Currently
     * consist of:
     *  0) Path to arguments .txt file
     *  1) Path to data .txt file
     *  2) Desired path to results .txt file
     *  3) Number of numberOfIterations
     *  4) Length of burin in
     *  5) Thinning - every nth iteration to save
     *  6) Every nth iteration to output feedback to terminal
     *  7) Which random number seed to use
     *  8) Whether to use likelihoodFamily selection (0/1)
     *  9) Whether to use alternative initial values (0/1)
     *  10) Number of likelihoodFamily space prior components
     *  11) Which likelihoodFamily space prior distribution to use (0=Poisson, 1=BetaBinomial)
     *  ... Either the Poisson likelihoodFamily prior rate(s) or the beta-binomial hyper-
     *  parameter(s)
     * See {@link Objects.Arguments#Arguments(java.lang.String[])} for which
     * fields these are saved into.
     * 
     * @throws IOException 
     */
    public static void main(String[] args) throws IOException {
        
        /*****************************************
         *** READ ARGUMENTS FILE AND DATA FILE ***
         *****************************************/
        
        // Read command line arguments and Arguments File
        Arguments arguments = new Arguments(args);

        // Create data object
        Data data = new Data(arguments);

        /*************************************************
         *** INITIALISE OBJECTS REQUIRED FOR MAIN LOOP ***
         *************************************************/
        
        // Initialise Random Number Generator object, and set up proposal distributions
        randomDraws = new Random(arguments.whichSeed);
        ProposalDistributions propSdsOb = new ProposalDistributions(arguments, data);

        // Initialise prior distributions
        Priors priors = new Priors(arguments, data);

        // Disable null moves for conjugate modelling without Tau       
        if (data.whichLikelihoodType==LikelihoodTypes.GAUSSIAN_CONJ.ordinal()|
                data.whichLikelihoodType==LikelihoodTypes.GAUSSIAN_MARGINAL_CONJ.ordinal()) {
            if (data.modelTau==0) {
                arguments.probRemove = 0.33;
                arguments.probAdd = 0.33;
                arguments.probSwap = 0.34;
                arguments.probNull = 0;                
            }
        }
        IterationValues curr = new IterationValues(arguments, data, priors);
        IterationValues prop = new IterationValues(arguments, data, priors);

        // Inititialise results file
        FileWriter results = new FileWriter(arguments.pathToResultsFile);
        BufferedWriter buffer = new BufferedWriter(results);
        buffer.write(
                "Likelihood"
                +" ModelSpacePriorFamily"
                +" V"
                +" startRJ"
                +" R"
                +" varsWithFixedPriors"
                +" nBetaHyperPriorComp"
                +" allModelScoresUpToDim"
                +" nRjComp"
                +" iterations"
                +" burnin"
                +" thin");
        buffer.newLine();   // Model space means (and splits)
        if (data.whichLikelihoodType==LikelihoodTypes.WEIBULL.ordinal()) {
            buffer.write("Weibull ");
        } else if (data.whichLikelihoodType==LikelihoodTypes.LOGISTIC.ordinal()) {
            buffer.write("Logistic ");
        } else if (data.whichLikelihoodType==LikelihoodTypes.GAUSSIAN.ordinal()) {
            buffer.write("Gaussian ");
        } else if (data.whichLikelihoodType==LikelihoodTypes.GAUSSIAN_CONJ.ordinal()) {
            buffer.write("GaussianConj ");
        } else if (data.whichLikelihoodType==LikelihoodTypes.GAUSSIAN_MARGINAL.ordinal()) {
            buffer.write("GaussianMarg ");
        } else if (data.whichLikelihoodType==LikelihoodTypes.GAUSSIAN_MARGINAL_CONJ.ordinal()) {
            buffer.write("GaussianMargConj ");
        }
        if (arguments.modelSpacePriorFamily==0) {
            buffer.write("Poisson");
        } else if (arguments.modelSpacePriorFamily==1) {
            buffer.write("BetaBinomial");
        }
        buffer.write(
                " "+data.totalNumberOfCovariates
                +" "+data.numberOfCovariatesToFixInModel
                +" "+0 // This was the number of clusters (R)
                +" "+data.numberOfCovariatesWithInformativePriors
                +" "+data.numberOfHierarchicalCovariatePriorPartitions
                +" "+data.allModelScoresUpToDim
                +" "+arguments.numberOfModelSpacePriorPartitions
                +" "+arguments.numberOfIterations
                +" "+arguments.burnInLength
                +" "+arguments.thinningInterval);
        buffer.newLine();   // Model space means (and splits)
        for (int c=0; c<arguments.numberOfModelSpacePriorPartitions; c++) {
            if (arguments.modelSpacePriorFamily==0) {
                buffer.write(data.modelSpacePoissonPriorMeans[c]+" ");
            } else if (arguments.modelSpacePriorFamily==1) {
                buffer.write(
                        arguments.modelSpaceBetaBinomialPriorHyperparameterA[c]+" "
                        +arguments.modelSpaceBetaBinomialPriorHyperparameterB[c]+" ");                
            }
        }
        if (arguments.numberOfModelSpacePriorPartitions>1) {
            for (int c=0; c<(arguments.numberOfModelSpacePriorPartitions+1); c++) {
                buffer.write(data.modelSpacePartitionIndices[c]+" ");
            }
        }
        
        /**********************************************************************
        *** All single SNP models *********************************************
        ***********************************************************************/
        
        int[] originalModel;
        if (data.allModelScoresUpToDim >= 1) {
            originalModel = curr.model;
            /**
             * Start with the null model
             */
            for (int v=data.numberOfCovariatesToFixInModel; v<data.totalNumberOfCovariates; v++) {
                curr.model[v]=0;
            }
            curr.modelDimension = GeneralMethods.countPresVars(data.totalNumberOfCovariates, curr.model);
            curr.modelSpacePartitionDimensions = GeneralMethods.countPresVarsComps(
                    arguments.numberOfModelSpacePriorPartitions, data.modelSpacePartitionIndices, curr.model);
            curr.conjugate_calcLogLike(arguments, data);
            buffer.newLine();
            buffer.write("Null "+curr.logLikelihood);
            /**
             * Calculate likelihood score for every model
             */
            for (int v=data.numberOfCovariatesToFixInModel; v<data.totalNumberOfCovariates; v++) {
                curr.model[v]=1;
                curr.modelDimension = GeneralMethods.countPresVars(data.totalNumberOfCovariates, curr.model);
                curr.modelSpacePartitionDimensions = GeneralMethods.countPresVarsComps(
                        arguments.numberOfModelSpacePriorPartitions, data.modelSpacePartitionIndices, curr.model);
                curr.conjugate_calcLogLike(arguments, data);
                buffer.newLine();
                buffer.write(data.covariateNames[v]+" "+curr.logLikelihood);
                curr.model[v]=0;
            }
            if (data.allModelScoresUpToDim >= 2) {
                /**
                 * Calculate likelihood score for every double model
                 */
                for (int v1=data.numberOfCovariatesToFixInModel; v1<(data.totalNumberOfCovariates-1); v1++) {
                    for (int v2=(v1+1); v2<data.totalNumberOfCovariates; v2++) {
                        curr.model[v1]=1;
                        curr.model[v2]=1;
                            curr.modelDimension = GeneralMethods.countPresVars(data.totalNumberOfCovariates, curr.model);
                            curr.modelSpacePartitionDimensions = GeneralMethods.countPresVarsComps(
                                    arguments.numberOfModelSpacePriorPartitions, data.modelSpacePartitionIndices, curr.model);
                            curr.conjugate_calcLogLike(arguments, data);
                            buffer.newLine();
                            buffer.write(data.covariateNames[v1]+"_AND_"+data.covariateNames[v2]+" "+curr.logLikelihood);
                        curr.model[v1]=0;
                        curr.model[v2]=0;
                    }
                }                
            }
            if (data.allModelScoresUpToDim >= 3) {
                /**
                 * Calculate likelihood score for every triple SNP model
                 */
                for (int v1=data.numberOfCovariatesToFixInModel; v1<(data.totalNumberOfCovariates-2); v1++) {
                    for (int v2=(v1+1); v2<(data.totalNumberOfCovariates-1); v2++) {
                        for (int v3=(v2+1); v3<data.totalNumberOfCovariates; v3++) {
                            curr.model[v1]=1;
                            curr.model[v2]=1;
                            curr.model[v3]=1;
                                curr.modelDimension = GeneralMethods.countPresVars(data.totalNumberOfCovariates, curr.model);
                                curr.modelSpacePartitionDimensions = GeneralMethods.countPresVarsComps(
                                        arguments.numberOfModelSpacePriorPartitions, data.modelSpacePartitionIndices, curr.model);
                                curr.conjugate_calcLogLike(arguments, data);
                                buffer.newLine();
                                buffer.write(data.covariateNames[v1]+"_AND_"+data.covariateNames[v2]+"_AND_"+data.covariateNames[v3]+" "+curr.logLikelihood);
                            curr.model[v1]=0;
                            curr.model[v2]=0;
                            curr.model[v3]=0;
                        }
                    }
                }                
            }
            if (data.allModelScoresUpToDim >= 4) {
                /**
                 * Calculate likelihood score for every quadruple SNP model
                 */
                for (int v1=data.numberOfCovariatesToFixInModel; v1<(data.totalNumberOfCovariates-3); v1++) {
                    for (int v2=(v1+1); v2<(data.totalNumberOfCovariates-2); v2++) {
                        for (int v3=(v2+1); v3<(data.totalNumberOfCovariates-1); v3++) {
                            for (int v4=(v3+1); v4<data.totalNumberOfCovariates; v4++) {
                                curr.model[v1]=1;
                                curr.model[v2]=1;
                                curr.model[v3]=1;
                                curr.model[v4]=1;
                                    curr.modelDimension = GeneralMethods.countPresVars(data.totalNumberOfCovariates, curr.model);
                                    curr.modelSpacePartitionDimensions = GeneralMethods.countPresVarsComps(
                                            arguments.numberOfModelSpacePriorPartitions, data.modelSpacePartitionIndices, curr.model);
                                    curr.conjugate_calcLogLike(arguments, data);
                                    buffer.newLine();
                                    buffer.write(data.covariateNames[v1]+"_AND_"+data.covariateNames[v2]+"_AND_"+data.covariateNames[v3]+"_AND_"+data.covariateNames[v4]+" "+curr.logLikelihood);
                                curr.model[v1]=0;
                                curr.model[v2]=0;
                                curr.model[v3]=0;
                                curr.model[v4]=0;
                            }
                        }
                    }
                }                
            }
            if (data.allModelScoresUpToDim >= 5) {
                /**
                 * Calculate likelihood score for every 5 SNP model
                 */
                for (int v1=data.numberOfCovariatesToFixInModel; v1<(data.totalNumberOfCovariates-4); v1++) {
                    for (int v2=(v1+1); v2<(data.totalNumberOfCovariates-3); v2++) {
                        for (int v3=(v2+1); v3<(data.totalNumberOfCovariates-2); v3++) {
                            for (int v4=(v3+1); v4<(data.totalNumberOfCovariates-1); v4++) {
                                for (int v5=(v4+1); v4<data.totalNumberOfCovariates; v5++) {
                                    curr.model[v1]=1;
                                    curr.model[v2]=1;
                                    curr.model[v3]=1;
                                    curr.model[v4]=1;
                                    curr.model[v5]=1;
                                        curr.modelDimension = GeneralMethods.countPresVars(data.totalNumberOfCovariates, curr.model);
                                        curr.modelSpacePartitionDimensions = GeneralMethods.countPresVarsComps(
                                                arguments.numberOfModelSpacePriorPartitions, data.modelSpacePartitionIndices, curr.model);
                                        curr.conjugate_calcLogLike(arguments, data);
                                        buffer.newLine();
                                        buffer.write(data.covariateNames[v1]+"_AND_"+data.covariateNames[v2]+"_AND_"+data.covariateNames[v3]+"_AND_"+data.covariateNames[v4]+"_AND_"+data.covariateNames[v5]+" "+curr.logLikelihood);
                                    curr.model[v1]=0;
                                    curr.model[v2]=0;
                                    curr.model[v3]=0;
                                    curr.model[v4]=0;
                                    curr.model[v5]=0;
                                }
                            }
                        }
                    }
                }                
            }
            /**
             * Set curr back to how it was
             */
            curr.model=originalModel;
            curr.modelDimension = GeneralMethods.countPresVars(data.totalNumberOfCovariates, curr.model);
            curr.modelSpacePartitionDimensions = GeneralMethods.countPresVarsComps(
                    arguments.numberOfModelSpacePriorPartitions, data.modelSpacePartitionIndices, curr.model);
            curr.conjugate_calcLogLike(arguments, data);
        }
        
        
        buffer.newLine(); // Variable names
        if (data.whichLikelihoodType==LikelihoodTypes.WEIBULL.ordinal()) {
            buffer.write("LogWeibullScale ");   // Weibull k                                
        } else if (data.whichLikelihoodType==LikelihoodTypes.GAUSSIAN.ordinal()|
                data.whichLikelihoodType==LikelihoodTypes.GAUSSIAN_MARGINAL.ordinal()) {
            buffer.write("LogGaussianResidual ");   // Gaussian residual            
        }
        buffer.write("alpha ");
        for (int v=0; v<data.totalNumberOfCovariates; v++) {
            buffer.write(data.covariateNames[v]+" ");
        }
        if (data.numberOfHierarchicalCovariatePriorPartitions>0) {
            for (int c=0; c<data.numberOfHierarchicalCovariatePriorPartitions; c++) {
                buffer.write("LogBetaPriorSd"+(c+1)+" ");   // beta hyper prior sds
            }
        }
        buffer.write("LogLikelihood ");   // log-Likelihood
        buffer.newLine();

        /**********************************************************************
        *** MAIN LOOP *********************************************************
        ***********************************************************************/

        int counter = 0;
        System.out.println("");
        System.out.println("--------------------------------");
        System.out.println("--- Initiating MCMC sampling ---");
        System.out.println("--------------------------------");
        long t1 = 0;
        long t2 = 0;
        for(int i=0; i<arguments.numberOfIterations; i++) {

            // update
            if (data.whichLikelihoodType==LikelihoodTypes.GAUSSIAN_MARGINAL_CONJ.ordinal()|
                    data.whichLikelihoodType==LikelihoodTypes.GAUSSIAN_CONJ.ordinal()
                    ) {
                prop.conjugate_update(arguments, data, curr, priors, propSdsOb, randomDraws);                
            } else {
                prop.update(arguments, data, curr, priors, propSdsOb, randomDraws);                
            }
            
            // Decide whether proposal is proposalAccepted and, if so, update 'current'
            // likelihoodFamily to proposal ADAPTION
            double accDraw = randomDraws.nextFloat();
            if (accDraw < prop.acceptanceProbability) {
                curr.setTo(prop);
                prop.proposalAccepted=1;
            } else {
                prop.setTo(curr);
                prop.proposalAccepted=0;
            }
            
            // Adapting
            if (i<=arguments.adaptionLength) {
                // Early feedback to get a sense of run time
                if (i==1) {
                    System.out.println("1 iteration complete");
                    t1 = System.currentTimeMillis();
                }
                if (i==10) {System.out.println("10 iterations complete");}
                if (i==100) {System.out.println("100 iterations complete");}
                if (i==1000) {
                    System.out.println("1000 iterations complete");
                    long minsFor1milIts = (System.currentTimeMillis()-t1)/(60);
                    System.out.println("Estimated minutes for 1 million iterations: "+minsFor1milIts);
                    System.out.println("------------------------------");
                }
                propSdsOb.adapt(data, prop, i);
                if (i==arguments.adaptionLength) {
                    System.out.println("Likelihood (usefull for debugging): "+curr.logLikelihood);
                }
            }
  
            // Write resulting 'current' likelihoodFamily to results file
            if (i>=arguments.burnInLength) {
                if (i % arguments.thinningInterval == 0) {
                    RegressionMethods.writeToResultsFile(arguments, data, curr, 
                            buffer);
                }
            }

            if (counter==arguments.consoleOutputInterval-1) {
                System.out.println((i+1)+" / "+arguments.numberOfIterations+" iterations " +
                        "complete");
//                System.out.println("Likelihood (for debugging) "+curr.logLikelihood);
                counter = 0;
            } else {counter++;}

        }

        // Close buffer that is used to write results
        buffer.close();
        System.out.println("------------------------------");
        System.out.println("--- MCMC sampling complete ---");
        System.out.println("------------------------------");
        System.out.println("");        
        System.out.println("Results written to "+arguments.pathToResultsFile );

    }

}


