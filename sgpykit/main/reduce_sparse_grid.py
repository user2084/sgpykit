import numpy as np

from sgpykit.src.detect_insufficient_tolerance import detect_insufficient_tolerance
from sgpykit.src.mysortrows import mysortrows
from sgpykit.util import matlab


def reduce_sparse_grid(S, tol=1e-14):
    """
    Reduce a sparse grid by removing duplicate points within a given tolerance.

    Given a sparse grid stored as a list of tensor grids, this function generates a
    reduced structure containing only non-repeated points and corresponding weights.

    Parameters
    ----------
    S : dict or object
        Sparse grid structure with fields:
        - knots: list of tensor grid knots
        - weights: list of corresponding weights
    tol : float, optional
        Tolerance to identify coincident points (default is 1e-14)

    Returns
    -------
    Sr : dict or object
        Reduced sparse grid structure with fields:
        - knots: list of non-repeated knots
        - weights: list of corresponding weights
        - size: number of non-repeated knots
        - n: forward map from [S.knots] to Sr.knots
        - m: inverse map from Sr.knots to [S.knots]
    """
    if isinstance(S.knots, list):
        knots_as_matrix = np.hstack(S.knots)  # TODO: check above might be not sufficient, need more like ismatrix(S.knots)
    else:
        knots_as_matrix = S.knots
    kk = np.atleast_2d(np.transpose(knots_as_matrix))
    ww = np.atleast_1d(np.squeeze(np.hstack(S.weights)))
    # first of all, a safety check: are we using a sufficiently fine tol when detecting identical points?
    detect_insufficient_tolerance(kk, tol)
    # first, I order the rows of kk in lexicographic order. If I use mysortrows then two rows like
    # [a b c d]
    # [a-t b c+t d]
    # are considered equal (if t < tol ) and therefore placed one after the other
    #[kk_ordered,i]=sortrows(kk);
    # i is an index vector that maps kk into kk_ordered, i.e. kk[i,:]==kk_ordered
    #[kk_ordered,i]=mysortrows(kk,tol/size(kk,2)); # this will not work properly if more than 10 rv are used, the
    #tol will be too small!
    kk_ordered,i = mysortrows(kk, tol)

    # next I take the difference of two consecutive rows. if the difference is small, then the rows are the same, i.e. the knot is the same
    dkk_ordered = np.diff(kk_ordered,1,0)

    # I measure the difference with infty norm instead of L2 norm:
    # i take  the maximum component of each row (2 means "operate on columns"):
    #       max(abs(dkk_ordered),[],2)
    # then I want to see which ones have this max bigger than tol
    #       j*=max(abs(dkk_ordered),[],2)>tol
    # this command returns a vector of True and false
    #       j*=[1 1 0 1 1]
    # this means that 2nd multiindex is different from 1st, 3rd from 2nd, but 4th is equal to 3rd ( diff(3)=v(4)-v(3) ). So I shouldn't take it.
    # How do I use this j* to build a vector j with the rows that I have to select from kk?
    # I could use only the rows of j* that are True
    #       j=find((j*==True))
    #       kk_reduced=kk_order[j,:],
    # NB
    #       I can shortcut find((j*==True)) with  find(j*)
    # so that I take rows 1,2, 4 but not 3.
    # If I do like this, I forget the last row!! (diff has 1 row less than kk). So I add a non zero difference at the end, and come to
    # matlab: j=find([max(abs(dkk_ordered),[],2)>tol;2*tol]);
    j = np.where(np.concatenate((np.max(np.abs(dkk_ordered), axis=1) > tol, [2 * tol])))[0]

    #j=find([sqrt(sum(dkk_ordered.^2,2))>tol;2*tol]); #L2 norm

    # now I can create the reduced knots
    kk_reduced = kk_ordered[j,:]
    # in the same way, I pick up only the j entries of the indeces vector i.
    index_reduced = np.transpose(i[j])

    # in this way I have a mapping from the original set kk to the reduced set kk_reduced:
    # i.e. kk[index_reduced,:]==kk_reduced.
    # this means that:
    #   -> lenght(index_reduced) < size(kk,1)
    #   -> index_reduced is not sorted

    # I also need the inverse mapping, i.e. how to reconstruct the original kk given kk_reduced. This is needed for the computation of weights
    # associated to each knot

    ### naif solution

    # invindex has to be of the same length than kk and kk_ordered.
    # NB: length(i)==size(kk,1) as it is the mapping from kk to kk_ordered

    invindex = np.zeros(len(i), dtype='int')

    k = 0

    # I build invindex working on kk_ordered: if for example
    # j=[1 2 4 5 7]
    # then it means that knot 1 will be picked up, and placed in position 1  in kk_reduced. Same for knot 2.
    # Conversely knot 3 won't be picked up, and 4 will be placed instead in position 3 ! Then it means that knot 3 and 4 will have a single
    # instance in kk_reduced, and the position on this image in kk_reduced is written in invindex. i.e. so far
    # invindex=[1 2 3 3 4 5 5].
    # I read this as "element in position k in kk_ordered becomes element invindex(k) in kk_reduced"

    for l in range(len(j)):
        invindex[k:j[l]+1] = l
        k = j[l] + 1

    # then I go back to the original kk. I go from kk to kk_ordered with the map i: kk[i,:]==kk_ordered.
    # So far invindex goes from kk_ordered to kk_reduced, and I would like to permute invindex with i^-1, to go back
    # to the order of kk:
    # invindex_kk=invindex(i^-1). (@)
    # I can do like this: say that
    # i=[3 1 2], invindex=[1 1 2]
    # then kk[3,:] is the 1st row of kk_ordered, and goes in the 1st row of kk_reduced. Then from position 1 of i and invindex I get
    # invindex_kk(3)=invindex(1)
    # and similar, 1st row of kk -> 2nd row of kk_ordered -> 1st row of kk_reduced, then
    # invindex_kk(1)=invindex(2)
    # therefore in general
    # invindex_kk(i)=invindex (that is, inverting @)<--------------- in short, invindex_kk permuted as i gives invindex
    # Then, instead of building two vectors, invindex and invindex_kk, i overwrite the former with the latter
    ## copy required otherwise overlapping memory access will lead to wrong results (dont: invindex[i] = invindex)
    tmp = invindex.copy()
    invindex[i] = tmp
    # The usage of invindex is:
    # "element in position k of kk goes in position invindex(k) in kk_reduced"

    nb_knots_reduced = len(index_reduced)
    # weights dtype follows ww, so complex-valued evaluations accumulate
    # correctly instead of being cast to real (ComplexWarning)
    weights = np.zeros(nb_knots_reduced, dtype=np.asarray(ww).dtype)
    Sr = matlab.struct(
        knots=np.transpose(kk_reduced),
        m=index_reduced, # 0-based index
        weights=weights,
        n=invindex, # 0-based index, finally, i use invindex to compute weights
        size=nb_knots_reduced
    )

    for j, idx_reduced in enumerate(invindex):
        Sr.weights[idx_reduced] += ww[j]

    return Sr
