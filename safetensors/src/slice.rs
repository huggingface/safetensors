//! Module handling lazy loading via iterating on slices on the original buffer.
use crate::tensor::TensorView;
use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

/// Error representing invalid slicing attempt
#[derive(Debug)]
pub enum InvalidSlice {
    /// When the client asked for more slices than the tensors has dimensions
    TooManySlices,
}

#[derive(Debug, Clone)]
/// Generic structure used to index a slice of the tensor
pub enum TensorIndexer {
    //Select(usize),
    /// This is a regular slice, purely indexing a chunk of the tensor
    Narrow(Bound<usize>, Bound<usize>),
    //IndexSelect(Tensor),
}

// impl From<usize> for TensorIndexer {
//     fn from(index: usize) -> Self {
//         TensorIndexer::Select(index)
//     }
// }

// impl From<&[usize]> for TensorIndexer {
//     fn from(index: &[usize]) -> Self {
//         let tensor = index.into();
//         TensorIndexer::IndexSelect(tensor)
//     }
// }
//
// impl From<Vec<usize>> for TensorIndexer {
//     fn from(index: Vec<usize>) -> Self {
//         let tensor = Tensor::of_slice(&index);
//         TensorIndexer::IndexSelect(tensor)
//     }
// }

macro_rules! impl_from_range {
    ($range_type:ty) => {
        impl From<$range_type> for TensorIndexer {
            fn from(range: $range_type) -> Self {
                use std::ops::Bound::*;

                let start = match range.start_bound() {
                    Included(idx) => Included(*idx),
                    Excluded(idx) => Excluded(*idx),
                    Unbounded => Unbounded,
                };

                let end = match range.end_bound() {
                    Included(idx) => Included(*idx),
                    Excluded(idx) => Excluded(*idx),
                    Unbounded => Unbounded,
                };

                TensorIndexer::Narrow(start, end)
            }
        }
    };
}

impl_from_range!(Range<usize>);
impl_from_range!(RangeFrom<usize>);
impl_from_range!(RangeFull);
impl_from_range!(RangeInclusive<usize>);
impl_from_range!(RangeTo<usize>);
impl_from_range!(RangeToInclusive<usize>);

/// Trait used to implement multiple signatures for ease of use of the slicing
/// of a tensor
pub trait IndexOp<'data, T> {
    /// Returns a slicing iterator which are the chunks of data necessary to
    /// reconstruct the desired tensor.
    fn slice(&'data self, index: T) -> Result<SliceIterator<'data>, InvalidSlice>;
}

impl<'data, A> IndexOp<'data, A> for TensorView<'data>
where
    A: Into<TensorIndexer>,
{
    fn slice(&'data self, index: A) -> Result<SliceIterator<'data>, InvalidSlice> {
        self.sliced_data(vec![index.into()])
    }
}

impl<'data, A> IndexOp<'data, (A,)> for TensorView<'data>
where
    A: Into<TensorIndexer>,
{
    fn slice(&'data self, index: (A,)) -> Result<SliceIterator<'data>, InvalidSlice> {
        let idx_a = index.0.into();
        self.sliced_data(vec![idx_a])
    }
}

impl<'data, A, B> IndexOp<'data, (A, B)> for TensorView<'data>
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
{
    fn slice(&'data self, index: (A, B)) -> Result<SliceIterator<'data>, InvalidSlice> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        self.sliced_data(vec![idx_a, idx_b])
    }
}

impl<'data, A, B, C> IndexOp<'data, (A, B, C)> for TensorView<'data>
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
    C: Into<TensorIndexer>,
{
    fn slice(&'data self, index: (A, B, C)) -> Result<SliceIterator<'data>, InvalidSlice> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        let idx_c = index.2.into();
        self.sliced_data(vec![idx_a, idx_b, idx_c])
    }
}

// impl<A, B, C, D> IndexOp<(A, B, C, D)> for TensorView<'data>
// where
//     A: Into<TensorIndexer>,
//     B: Into<TensorIndexer>,
//     C: Into<TensorIndexer>,
//     D: Into<TensorIndexer>,
// {
//     fn slice(&self, index: (A, B, C, D)) -> TensorView<'data> {
//         let idx_a = index.0.into();
//         let idx_b = index.1.into();
//         let idx_c = index.2.into();
//         let idx_d = index.3.into();
//         self.sliced_data(&[idx_a, idx_b, idx_c, idx_d])
//     }
// }
//
// impl<A, B, C, D, E> IndexOp<(A, B, C, D, E)> for TensorView<'data>
// where
//     A: Into<TensorIndexer>,
//     B: Into<TensorIndexer>,
//     C: Into<TensorIndexer>,
//     D: Into<TensorIndexer>,
//     E: Into<TensorIndexer>,
// {
//     fn slice(&self, index: (A, B, C, D, E)) -> TensorView<'data> {
//         let idx_a = index.0.into();
//         let idx_b = index.1.into();
//         let idx_c = index.2.into();
//         let idx_d = index.3.into();
//         let idx_e = index.4.into();
//         self.sliced_data(&[idx_a, idx_b, idx_c, idx_d, idx_e])
//     }
// }
//
// impl<A, B, C, D, E, F> IndexOp<(A, B, C, D, E, F)> for TensorView<'data>
// where
//     A: Into<TensorIndexer>,
//     B: Into<TensorIndexer>,
//     C: Into<TensorIndexer>,
//     D: Into<TensorIndexer>,
//     E: Into<TensorIndexer>,
//     F: Into<TensorIndexer>,
// {
//     fn slice(&self, index: (A, B, C, D, E, F)) -> TensorView<'data> {
//         let idx_a = index.0.into();
//         let idx_b = index.1.into();
//         let idx_c = index.2.into();
//         let idx_d = index.3.into();
//         let idx_e = index.4.into();
//         let idx_f = index.5.into();
//         self.sliced_data(&[idx_a, idx_b, idx_c, idx_d, idx_e, idx_f])
//     }
// }
//
// impl<A, B, C, D, E, F, G> IndexOp<(A, B, C, D, E, F, G)> for TensorView<'data>
// where
//     A: Into<TensorIndexer>,
//     B: Into<TensorIndexer>,
//     C: Into<TensorIndexer>,
//     D: Into<TensorIndexer>,
//     E: Into<TensorIndexer>,
//     F: Into<TensorIndexer>,
//     G: Into<TensorIndexer>,
// {
//     fn slice(&self, index: (A, B, C, D, E, F, G)) -> TensorView<'data> {
//         let idx_a = index.0.into();
//         let idx_b = index.1.into();
//         let idx_c = index.2.into();
//         let idx_d = index.3.into();
//         let idx_e = index.4.into();
//         let idx_f = index.5.into();
//         let idx_g = index.6.into();
//         self.sliced_data(&[idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g])
//     }
// }

/// Iterator used to return the bits of the overall tensor buffer
/// when client asks for a slice of the original tensor.
pub struct SliceIterator<'data> {
    view: &'data TensorView<'data>,
    indices: Vec<(usize, usize)>,
    newshape: Vec<usize>,
}

impl<'data> SliceIterator<'data> {
    pub(crate) fn new(
        view: &'data TensorView<'data>,
        slices: Vec<TensorIndexer>,
    ) -> Result<Self, InvalidSlice> {
        // Make sure n. axis does not exceed n. of dimensions
        let n_slice = slices.len();
        let n_shape = view.shape().len();
        if n_slice > n_shape {
            return Err(InvalidSlice::TooManySlices);
        }
        let mut newshape = Vec::with_capacity(view.shape().len());

        // Minimum span is the span of 1 item;
        let mut span = view.dtype().size();
        let mut indices = vec![];
        // Everything is row major.
        for (i, &shape) in view.shape().iter().enumerate().rev() {
            if i >= slices.len() {
                // We are  not slicing yet, just increase the local span
                newshape.push(shape);
            } else {
                let slice = &slices[i];
                let (start, stop) = match slice {
                    TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded) => (0, shape),
                    TensorIndexer::Narrow(Bound::Unbounded, Bound::Excluded(stop)) => (0, *stop),
                    TensorIndexer::Narrow(Bound::Unbounded, Bound::Included(stop)) => {
                        (0, *stop + 1)
                    }
                    TensorIndexer::Narrow(Bound::Included(s), Bound::Unbounded) => (*s, shape),
                    TensorIndexer::Narrow(Bound::Included(s), Bound::Excluded(stop)) => (*s, *stop),
                    TensorIndexer::Narrow(Bound::Included(s), Bound::Included(stop)) => {
                        (*s, *stop + 1)
                    }
                    TensorIndexer::Narrow(Bound::Excluded(s), Bound::Unbounded) => (*s + 1, shape),
                    TensorIndexer::Narrow(Bound::Excluded(s), Bound::Excluded(stop)) => {
                        (*s + 1, *stop)
                    }
                    TensorIndexer::Narrow(Bound::Excluded(s), Bound::Included(stop)) => {
                        (*s + 1, *stop + 1)
                    }
                };
                newshape.push(stop - start);
                if indices.is_empty() {
                    if start == 0 && stop == shape {
                        // We haven't started to slice yet, just increase the span
                    } else {
                        let offset = start * span;
                        let small_span = stop * span - offset;
                        indices.push((offset, offset + small_span));
                    }
                } else {
                    let mut newindices = vec![];
                    for n in start..stop {
                        let offset = n * span;
                        for (old_start, old_stop) in &indices {
                            newindices.push((old_start + offset, old_stop + offset));
                        }
                    }
                    indices = newindices;
                }
            }
            span *= shape;
        }
        if indices.is_empty() {
            indices.push((0, view.data().len()));
        }
        // Reversing so we can pop faster while iterating on the slice
        let indices = indices.into_iter().rev().collect();
        let newshape = newshape.into_iter().rev().collect();
        Ok(Self {
            view,
            indices,
            newshape,
        })
    }

    /// Gives back the amount of bytes still being in the iterator
    pub fn remaining_byte_len(&self) -> usize {
        self.indices
            .iter()
            .map(|(start, stop)| (stop - start))
            .sum()
    }

    /// Gives back the amount of bytes still being in the iterator
    pub fn newshape(&self) -> Vec<usize> {
        self.newshape.clone()
    }
}

impl<'data> Iterator for SliceIterator<'data> {
    type Item = &'data [u8];

    fn next(&mut self) -> Option<Self::Item> {
        // TODO We might want to move the logic from `new`
        // here actually to remove the need to get all the indices
        // upfront.
        let (start, stop) = self.indices.pop()?;
        Some(&self.view.data()[start..stop])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dtype, TensorView};

    #[test]
    fn test_helpers() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![1, 2, 3], &data).unwrap();

        let iterator = SliceIterator::new(
            &attn_0,
            vec![TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded)],
        )
        .unwrap();
        assert_eq!(iterator.remaining_byte_len(), 24);
        assert_eq!(iterator.newshape(), vec![1, 2, 3]);

        let iterator = SliceIterator::new(
            &attn_0,
            vec![
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1)),
            ],
        )
        .unwrap();
        assert_eq!(iterator.remaining_byte_len(), 12);
        assert_eq!(iterator.newshape(), vec![1, 1, 3]);
    }

    #[test]
    fn test_dummy() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![1, 2, 3], &data).unwrap();

        let mut iterator = SliceIterator::new(
            &attn_0,
            vec![TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded)],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            vec![
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            vec![
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            vec![
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        assert!(SliceIterator::new(
            &attn_0,
            vec![
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
            ],
        )
        .is_err(),);
    }

    #[test]
    fn test_slice_variety() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![1, 2, 3], &data).unwrap();

        let mut iterator = SliceIterator::new(
            &attn_0,
            vec![TensorIndexer::Narrow(
                Bound::Included(0),
                Bound::Excluded(1),
            )],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..24]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            vec![
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1)),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..12]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            vec![
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1)),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[0..4]));
        assert_eq!(iterator.next(), Some(&data[12..16]));
        assert_eq!(iterator.next(), None);

        let mut iterator = SliceIterator::new(
            &attn_0,
            vec![
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
                TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(2)),
                TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(1)),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[12..16]));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_slice_variety2() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let attn_0 = TensorView::new(Dtype::F32, vec![2, 3], &data).unwrap();

        let mut iterator = SliceIterator::new(
            &attn_0,
            vec![
                TensorIndexer::Narrow(Bound::Unbounded, Bound::Unbounded),
                TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(3)),
            ],
        )
        .unwrap();
        assert_eq!(iterator.next(), Some(&data[4..12]));
        assert_eq!(iterator.next(), Some(&data[16..24]));
        assert_eq!(iterator.next(), None);
    }
}
