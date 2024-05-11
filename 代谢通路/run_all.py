'''
Aim:
    Reproduce https://github.com/Angione-Lab/Synechococcus7002-metabolic-modelling in Python

Dependencies:
    conda create -n fba python=3.9
	conda activate fba
	conda install numpy
	conda install pandas
	conda install openpyxl
    conda install cobra
	conda install -c conda-forge cvxpy
	
Inputs:
    modelXML.xml
    bounds.xlsx
    
	Dataset1.xlsx
	Dataset2.xlsx
	
	
Ground Truths:
    transcriptsnew.csv
	all_atp_flux.csv
	all_ATPTF.csv
	
Output:
    my_transcripts.csv
	my_atp_flux.csv
	my_ATPTF.csv
'''

from pathlib import Path
import numpy as np
import pandas as pd
import cvxpy
import cobra
from cobra.io import read_sbml_model
from cobra.util import create_stoichiometric_matrix
import argparse
import mlflow
import os

def read_fbamodel_in_xml(xml_path):
    # modelXML.xml: the "M_photon_c" has been replaced into "M_photon_b".
    fbamodel = read_sbml_model(str(xml_path.resolve()))
    # remove empty or unannotated reactions and metabolites.
    reactions_to_remove = fbamodel.reactions[:57]
    fbamodel.remove_reactions(reactions_to_remove)
    metabolites_to_remove = [met for met in fbamodel.metabolites if met.id.endswith("_b")]
    fbamodel.remove_metabolites(metabolites_to_remove)
    assert len(fbamodel.reactions) == 742, 'Wrong number of reactions!'
    assert len(fbamodel.metabolites) == 696, 'Wrong number of metabolites!'
    assert len(fbamodel.genes) == 728, 'Wrong number of genes!'
    return fbamodel

def excel2fc(excel_file):
    dataset = pd.read_excel(excel_file, engine = 'openpyxl')
    locusTag = dataset.iloc[:, 0].tolist()
    rpkm_columns = [col for col in dataset.columns if 'RPKM' in col]
    datarpkm = dataset[rpkm_columns].to_numpy()
    controlmeans = datarpkm[:,:3].mean(axis=1)
    mask0 = controlmeans == 0
    controlmeans[mask0] = 1
    newfc = datarpkm[:, 3:] / controlmeans[:, np.newaxis]
    newfc[mask0] = 0
    return newfc, locusTag

def check_transcripts(transcripts,true_file):
	transcripts = np.roll(transcripts, shift = -1, axis = 1)
	transcripts = np.transpose(transcripts)
	true_trans = np.genfromtxt(true_file, delimiter=',')
	fro_err = np.linalg.norm(transcripts - true_trans, 'fro')
	print('The error of transcripts data =', fro_err)
	np.savetxt("my_transcripts.csv", transcripts, delimiter=",")

def substitute(stringa, i):
	# sub "x and y" into "min(x,y)"
    # sub "x or y" into "max(x,y)"
	i = i + 1 # skip the first space
	trovato_and = 1 if stringa[i] == 'a' else 0 # flag for "and" "or"
	
    # find the beginning of term x from the position i
	parentesi_trovate = 0
	j = i
	while (stringa[j] != '(' or parentesi_trovate != -1) and (stringa[j] != ',' or parentesi_trovate != 0) and (j > 0):
		j = j - 1 # pointer j moves forward
		if stringa[j] == ')':
			parentesi_trovate = parentesi_trovate + 1
		if stringa[j] == '(':
			parentesi_trovate = parentesi_trovate - 1

	if parentesi_trovate == -1 or stringa[j] == ',':
		j = j + 1

    # find the end of term y from the position of i
	if trovato_and == 1:
		k = i + 3 # " and "
	else:
		k = i + 2 # " or "

	parentesi_trovate = 0
	while (stringa[k] != ')' or parentesi_trovate != -1) and (stringa[k] != ',' or parentesi_trovate != 0) and (k < len(stringa)-1):
		k = k + 1
		if stringa[k] == '(':
			parentesi_trovate = parentesi_trovate + 1
		if stringa[k] == ')':
			parentesi_trovate = parentesi_trovate - 1

	if parentesi_trovate == -1 or stringa[k] == ',':
		k = k - 1

	parentesi_trovate = 0
	if trovato_and == 1:
		stringa_new = stringa[:j] + f"min({stringa[j:i]}, {stringa[i+4:k+1]})" + stringa[k+1:]
	else:
		stringa_new = stringa.replace(stringa[j:k+1], f"max({stringa[j:i]}, {stringa[i+3:k+1]})")

	return stringa_new

def associate_genes_reactions(stringa):
	if stringa != '':
		while ' or ' in stringa:
			i = 0
			while stringa[i:i+4] != ' or ':
				i += 1
			stringa = substitute(stringa, i)
		while ' and ' in stringa:
			i = 0
			while stringa[i:i+5] != ' and ':
				i += 1
			stringa = substitute(stringa, i)
		stringa_output = stringa.replace(" ", "")
	else:
		stringa_output = stringa
	return stringa_output

def solveQP_cvxpy(S, b, lb, ub):
    n = S.shape[1]
    v = cvxpy.Variable(n)
    objective = cvxpy.Minimize(1e-6 * 0.5 * cvxpy.sum_squares(v))
    constraints = [S @ v == b, lb <= v, v <= ub]
    problem = cvxpy.Problem(objective, constraints)
    optimal_value = problem.solve(verbose=False, solver='CLARABEL')
    # solver = ['CLARABEL', 'CPLEX', 'ECOS', 'ECOS_BB', 'OSQP', 'SCIPY', 'SCS']
    return optimal_value, v.value

def solveLP(c, S, w, lb, ub):
    x = cvxpy.Variable(len(c))
    objective = cvxpy.Maximize(c.T @ x)
    constraints = [S @ x == w, lb <= x, x <= ub]
    problem = cvxpy.Problem(objective, constraints)
    optimal_value = problem.solve(verbose=False, solver='SCIPY')
    # optimal_x = x.value
    return optimal_value

def flux_balance_minNorm(fbamodel,g):
	# Solve Linear Programming
    nrxn = len(fbamodel.reactions)
    nmet = len(fbamodel.metabolites)
    # Get the lower and upper bounds
    lb = np.array([rxn.lower_bound for rxn in fbamodel.reactions])
    ub = np.array([rxn.upper_bound for rxn in fbamodel.reactions])
    # B = pd.DataFrame({'rxnid': [rxn.id for rxn in fbamodel.reactions],
	# 				  'lower': lb,
	# 				  'upper': ub
    #                   })
    # B.to_csv('out_bounds.csv')

    # Primary Objective
    primaryObjID = 734  # fbamodel.reactions[734] = Reaction NEWBIOMASSCLIMITED
    c1 = np.zeros(nrxn)
    c1[primaryObjID] = 1.0
    S1 = np.array(create_stoichiometric_matrix(fbamodel))
    S1[8,734] = 45.7318
    w1 = np.zeros(nmet)
    vbiomass = solveLP(c1, S1, w1, lb, ub)
    print('Flux balance Biomass flux (%s):\t%f' % (fbamodel.reactions[primaryObjID].id, vbiomass))
	
    # Secondary Objective
    c2 = np.zeros(nrxn)
    c2[g] = 1.0 # atp = 69, photo1 = 697, photo2 = 696
    S2 = np.vstack((S1, c1))
    w2 = np.zeros(nmet+1)
    w2[-1] = vbiomass
    fmax = solveLP(c2, S2, w2, lb, ub)
    print('Flux balance secondary flux (%s):\t\t%f' % (fbamodel.reactions[g].id, fmax))
	
    # Solve Quadratic Programming
    S3 = np.vstack((S2, c2))
    w3 = np.zeros(nmet+2)
    w3[-2] = vbiomass
    w3[-1] = fmax
    # np.savetxt("out_S3.csv", S3, delimiter=",")
    # np.savetxt("out_w3.csv", w3, delimiter=",")
    obj_cvxpy, v_cvxpy = solveQP_cvxpy(S3, w3, lb, ub)
    print('quadratic opt_objfunc(cvxpy):  %f' % obj_cvxpy)
    return v_cvxpy

def set_expression_bounds(fbamodel, transcripts, condix, geni, reaction_expressions, ixs_geni_sorted_by_length, pos_genes_in_react_expr):
    yt = transcripts[:,condix-1]
    yt = yt[:, np.newaxis]
    eval_reaction_expression = reaction_expressions.copy()
    for i in range(len(geni)):
        yexp = yt[ixs_geni_sorted_by_length[i]]
        pos_gene1 = pos_genes_in_react_expr[i]
        for j in range(len(pos_gene1)):
            eval_reaction_expression[pos_gene1[j]] = eval_reaction_expression[pos_gene1[j]].replace(geni[i], format(yexp.item(0), ".15f"))
    eval_reaction_expression = ['1.0' if expr == '' else expr for expr in eval_reaction_expression]
    num_reaction_expression = [eval(e) for e in eval_reaction_expression]
    for i in range(len(reaction_expressions)):
        fbamodel.reactions[i].lower_bound = fbamodel.reactions[i].lower_bound * (num_reaction_expression[i] ** 3.5)
        fbamodel.reactions[i].upper_bound = fbamodel.reactions[i].upper_bound * (num_reaction_expression[i] ** 3.5)
    return fbamodel

def set_external_bounds(fbamodel, condix, excel_bounds):
    exbounds = pd.read_excel(excel_bounds, engine = 'openpyxl')
    for row in exbounds.iloc[:15].itertuples():
        fbamodel.reactions.get_by_id(row.reaction_id).lower_bound = getattr(row, f'new_val_{condix}')
    for row in exbounds.iloc[-2:].itertuples():
        fbamodel.reactions.get_by_id(row.reaction_id).upper_bound = getattr(row, f'new_val_{condix}')
    return fbamodel

def run_fba(fbamodel,g,transcripts, geni, reaction_expressions, \
			ixs_geni_sorted_by_length, pos_genes_in_react_expr, bounds_xlsx):
    nrxn = len(fbamodel.reactions)
    Vcvxpy = np.empty((nrxn, 0))
    for condix in range(1,25): # 24 experimental conditions: 1 ~ 24
        #print(condix)
        fbamodel1 = fbamodel.copy()
		# Read external values of flux bounds
        fbamodel1 = set_external_bounds(fbamodel1, condix, bounds_xlsx)
        # Set lower/upper bounds using transcriptome data
        fbamodel1 = set_expression_bounds(fbamodel1, transcripts, condix, geni, reaction_expressions, \
										  ixs_geni_sorted_by_length, pos_genes_in_react_expr)
        # Flux balance analysis
        v_cvxpy = flux_balance_minNorm(fbamodel1,g)
        v_cvxpy = v_cvxpy.squeeze()
        Vcvxpy = np.concatenate((Vcvxpy, v_cvxpy[:, np.newaxis]), axis = 1)
    Vcvxpy = np.roll(Vcvxpy, shift = -1, axis = 1)
    Vcvxpy = np.transpose(Vcvxpy)
    Vcvxpy = np.abs(Vcvxpy)
    Vcvxpy[Vcvxpy <= 0.0001] = 0
    return Vcvxpy

def create_multiomics(all_flux,transcripts):
    control_flux = all_flux[23, :]
    control_flux_no_zeros = np.where(control_flux == 0, 1, control_flux)
    ATP_FC = all_flux[:23, :] / control_flux_no_zeros
    ATP_FC[(control_flux == 0) & (ATP_FC == 0)] = 1
    max_val = ATP_FC.max()
    ATP_FC[(control_flux == 0) & (all_flux[:23, :] != 0)] = max_val
    ATP_FC[ATP_FC <= 0.0001] = 0
    ATPTF = np.hstack((np.transpose(transcripts[:,1:]), ATP_FC))
    ATPTF = np.vstack((ATPTF, np.ones((1, ATPTF.shape[1]))))
    return ATPTF

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset1", help="dataset file name in data dir", type=str, default="Dataset1.xlsx")
    parser.add_argument("--dataset2", help="dataset file name in data dir", type=str, default="Dataset2.xlsx")
    return parser.parse_known_args()[0]

def main():
    
    try:
        import pandas as pd
    except:
        os.system('pip uninstall pandas -y')
        os.system('pip install pandas')

    """Main entry."""
    xml_path = Path(".") / "modelXML.xml"
    fbamodel = read_fbamodel_in_xml(xml_path)
	
    # Preparation of transcriptomic data
    [newFC1, gexp1] = excel2fc("../data/"+args.dataset1)
    [newFC2, gexp2] = excel2fc("../data/"+args.dataset2)
    assert gexp1 == gexp2, "The gene name lists are not equal in Dataset 1 and 2."
    gexp = gexp1 # gene name list in expression data
    transcripts = np.hstack((np.ones((len(gexp),1)), newFC1, newFC2)) # shape = 3187 * 24, col 1 = ctrl
    check_transcripts(transcripts, 'transcriptsnew.csv')

    with mlflow.start_run(run_name="fba") as run:
        # compute_reaction_expression.m
        grules = [reaction.gene_reaction_rule for reaction in fbamodel.reactions]
        # formatting reaction_expressions
        reaction_expressions = [associate_genes_reactions(grule) for grule in grules]

        # Get ixs_geni_sorted_by_length, pos_genes_in_react_expr
        # ixs_geni_sorted_by_length is the index of geni[i] in gexp
        geni = [gene.name for gene in fbamodel.genes] # 728
        ixs_geni_sorted_by_length = [gexp.index(g) if g in gexp else None for g in geni]
        # pos_genes_in_react_expr is the indices of geni at reaction_expressions
        pos_genes_in_react_expr = [([k for k, expr in enumerate(reaction_expressions) if g in expr] or None) for g in geni]	

        # Run flux balance analysis
        # gindex is the index of the reaction used as secondary objective function
        # Possible gindex: atp = 69, photo1 = 697, photo2 = 696
        # input:
        gindex = 69
        fout_flux = 'my_atp_flux.csv'
        true_flux = 'all_atp_flux.csv'
        fout_fexp = 'my_ATPTF.csv'
        true_fexp = 'all_ATPTF.csv'

        Vcvxpy = run_fba(fbamodel.copy(),gindex,transcripts, geni, reaction_expressions, 
                        ixs_geni_sorted_by_length, pos_genes_in_react_expr, 'bounds.xlsx') # bound col 1 = ctrl
        np.savetxt(fout_flux, Vcvxpy, delimiter=",")
        # check
        true_atp_flux = np.genfromtxt(true_flux, delimiter=',')
        print('The flux error is', np.mean(np.abs(true_atp_flux - Vcvxpy)))
        
        # create multiomics data: transcripts + fluxes
        ATPTF = create_multiomics(Vcvxpy,transcripts)
        np.savetxt(fout_fexp, ATPTF, delimiter=",")
        # check
        trueATPTF = np.genfromtxt(true_fexp, delimiter=',')
        print('The flux + transcriptome error is', np.mean(np.abs(trueATPTF - ATPTF)))
        
        mlflow.log_artifact(local_path='my_atp_flux.csv')
        mlflow.log_artifact(local_path='my_ATPTF.csv')
        mlflow.log_artifact(local_path='my_transcripts.csv')

if __name__ == '__main__':
    main()
